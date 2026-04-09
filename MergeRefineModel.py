import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryWarpRegressor(nn.Module):
    """
    Input:
        left_pred  : [B, 1, H, K]
        left_recon : [B, 1, H, K]
        top_pred   : [B, 1, K, W]
        top_recon  : [B, 1, K, W]
        predictor  : [B, C, H, W]

    Output:
        warped predictor
        dense flow [B, H, W, 2]
        boundary-only shifts
    """

    def __init__(
        self,
        band_size=3,
        hidden=64,
        num_layers=4,
        max_shift_x=0.75,
        max_shift_y=0.75,
    ):
        super().__init__()
        self.band = band_size
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y

        layers = []
        in_ch = 6  # [pred_3, recon_3]
        ch = hidden
        layers.append(nn.Conv1d(in_ch, ch, kernel_size=3, padding=1))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(ch, ch, kernel_size=3, padding=1))
            layers.append(nn.GELU())

        self.backbone = nn.Sequential(*layers)

        # left sequence head: outputs K*2 per row-token
        self.left_head = nn.Conv1d(ch, band_size * 2, kernel_size=1)

        # top sequence head: outputs K*2 per col-token
        self.top_head = nn.Conv1d(ch, band_size * 2, kernel_size=1)

        # optional confidence / enable strength
        self.gate_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(ch, 1, kernel_size=1),
        )

    @staticmethod
    def make_base_grid(h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij"
        )
        return torch.stack([xx, yy], dim=-1)  # [H,W,2]

    @staticmethod
    def warp(x, flow):
        """
        x    : [B,C,H,W]
        flow : [B,H,W,2] in pixel units
        """
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        base_grid = BoundaryWarpRegressor.make_base_grid(h, w, device, dtype)
        base_grid = base_grid.unsqueeze(0).expand(b, h, w, 2).contiguous()

        norm_dx = 2.0 * flow[..., 0] / max(w - 1, 1)
        norm_dy = 2.0 * flow[..., 1] / max(h - 1, 1)

        grid = base_grid.clone()
        grid[..., 0] += norm_dx
        grid[..., 1] += norm_dy

        warped = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        return warped

    def build_feature_sequence(self, left_pred, left_recon, top_pred, top_recon):
        """
        Build [B, 6, H+W]
        left side: per row, 6-dim = [pred3, recon3]
        top side : per col, 6-dim = [pred3, recon3]
        """
        b, c, h, k = left_pred.shape
        _, _, k2, w = top_pred.shape
        assert c == 1, "현재 코드는 입력 boundary feature를 1채널 기준으로 작성"
        assert k == self.band and k2 == self.band

        # left: [B,1,H,K] -> [B,H,K]
        lp = left_pred[:, 0]    # [B,H,K]
        lr = left_recon[:, 0]   # [B,H,K]

        # top : [B,1,K,W] -> [B,W,K]
        tp = top_pred[:, 0].permute(0, 2, 1).contiguous()   # [B,W,K]
        tr = top_recon[:, 0].permute(0, 2, 1).contiguous()  # [B,W,K]

        # per token feature dim = 2K = 6
        left_feat = torch.cat([lp, lr], dim=-1)  # [B,H,6]
        top_feat = torch.cat([tp, tr], dim=-1)   # [B,W,6]

        seq = torch.cat([left_feat, top_feat], dim=1)   # [B,H+W,6]
        seq = seq.permute(0, 2, 1).contiguous()         # [B,6,H+W]
        return seq

    def build_dense_flow(self, left_shift, top_shift, h, w, device, dtype):
        """
        left_shift:
            dx_left [B,H,K], dy_left [B,H,K]
        top_shift:
            dx_top  [B,K,W], dy_top  [B,K,W]

        Returns:
            dense flow [B,H,W,2], only left/top bands are nonzero.
            overlap(top-left corner) is averaged.
        """
        dx_left, dy_left = left_shift
        dx_top, dy_top = top_shift

        b = dx_left.shape[0]
        k = self.band

        flow_x = torch.zeros((b, h, w), device=device, dtype=dtype)
        flow_y = torch.zeros((b, h, w), device=device, dtype=dtype)
        weight = torch.zeros((b, h, w), device=device, dtype=dtype)

        # left band
        flow_x[:, :, :k] += dx_left
        flow_y[:, :, :k] += dy_left
        weight[:, :, :k] += 1.0

        # top band
        flow_x[:, :k, :] += dx_top
        flow_y[:, :k, :] += dy_top
        weight[:, :k, :] += 1.0

        weight = torch.clamp(weight, min=1.0)
        flow_x = flow_x / weight
        flow_y = flow_y / weight

        flow = torch.stack([flow_x, flow_y], dim=-1)  # [B,H,W,2]
        return flow

    def forward(self, predictor, left_pred, left_recon, top_pred, top_recon):
        """
        predictor  : [B,C,H,W]
        left_pred  : [B,1,H,K]
        left_recon : [B,1,H,K]
        top_pred   : [B,1,K,W]
        top_recon  : [B,1,K,W]
        """
        b, c, h, w = predictor.shape
        device, dtype = predictor.device, predictor.dtype
        k = self.band

        seq = self.build_feature_sequence(left_pred, left_recon, top_pred, top_recon)  # [B,6,H+W]
        feat = self.backbone(seq)  # [B,hidden,H+W]

        # split tokens
        feat_left = feat[:, :, :h]     # [B,hidden,H]
        feat_top = feat[:, :, h:h+w]   # [B,hidden,W]

        # gate
        gate = torch.sigmoid(self.gate_head(feat)).view(b, 1, 1)  # [B,1,1]

        # left output: [B,2K,H] -> [B,H,K,2]
        left_raw = self.left_head(feat_left).permute(0, 2, 1).contiguous()
        left_raw = left_raw.view(b, h, k, 2)

        # top output: [B,2K,W] -> [B,W,K,2] -> [B,K,W,2]
        top_raw = self.top_head(feat_top).permute(0, 2, 1).contiguous()
        top_raw = top_raw.view(b, w, k, 2).permute(0, 2, 1, 3).contiguous()

        # bounded regression
        dx_left = torch.tanh(left_raw[..., 0]) * self.max_shift_x * gate
        dy_left = torch.tanh(left_raw[..., 1]) * self.max_shift_y * gate

        dx_top = torch.tanh(top_raw[..., 0]) * self.max_shift_x * gate
        dy_top = torch.tanh(top_raw[..., 1]) * self.max_shift_y * gate

        flow = self.build_dense_flow(
            left_shift=(dx_left, dy_left),
            top_shift=(dx_top, dy_top),
            h=h,
            w=w,
            device=device,
            dtype=dtype,
        )

        warped = self.warp(predictor, flow)

        return {
            "warped": warped,
            "flow": flow,
            "dx_left": dx_left,
            "dy_left": dy_left,
            "dx_top": dx_top,
            "dy_top": dy_top,
            "gate": gate.squeeze(-1),
        }
        























import torch

# a: (h+3, w+3)
# b: (h, w)

a_part = torch.cat([
    a[3:, :3],      # (h, 3)
    a[:3, 3:].T     # (w, 3)
], dim=0)           # (h+w, 3)

b_part = torch.cat([
    b[:, :3],       # (h, 3)
    b[:3, :].T      # (w, 3)
], dim=0)           # (h+w, 3)

out = torch.cat([a_part, b_part], dim=1)   # (h+w, 6)



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MergeRefineOracle4Param(nn.Module):
    """
    각 class는 4개 파라미터를 가짐:
      ax, ay : x/y 기본 이동량 (pixel unit)
      cx, cy : center participation (sigmoid로 0~1)

    최종 flow는:
      dx(x,y) = ax * (cx + (1-cx) * decay(x,y))
      dy(x,y) = ay * (cy + (1-cy) * decay(x,y))
    """

    def __init__(self):
        super().__init__()
        self.num_classes = 9

        # 초기화 예시
        # [ax, ay, cx, cy]
        init = torch.tensor([
            [ 0.00,  0.00,  8.0,  8.0],   # 0: no-op  (sigmoid(8) ~ 1)
            [ 0.25,  0.00, -8.0, -8.0],   # 1: x+
            [-0.25,  0.00, -8.0, -8.0],   # 2: x-
            [ 0.00,  0.25, -8.0, -8.0],   # 3: y+
            [ 0.00, -0.25, -8.0, -8.0],   # 4: y-
            [ 0.25,  0.25, -8.0, -8.0],   # 5: diag ++
            [-0.25, -0.25, -8.0, -8.0],   # 6: diag --
            [ 0.00,  0.25,  0.0, -2.0],   # 7: top-ish vertical
            [ 0.25,  0.00, -2.0,  0.0],   # 8: left-ish horizontal
        ], dtype=torch.float32)

        self.class_params = nn.Parameter(init)  # [9,4]

    @staticmethod
    def make_decay_map(h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij"
        )
        # top-left=1, bottom-right=0
        decay = 1.0 - 0.5 * (xx + yy)
        return decay.clamp(0.0, 1.0)

    @staticmethod
    def make_base_grid(h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij"
        )
        return torch.stack([xx, yy], dim=-1)  # [H,W,2]

    def build_flow_from_params(self, params, h, w, device, dtype):
        ax, ay, cx_raw, cy_raw = params.unbind(dim=0)

        cx = torch.sigmoid(cx_raw)
        cy = torch.sigmoid(cy_raw)

        decay = self.make_decay_map(h, w, device, dtype)  # [H,W]

        wx = cx + (1.0 - cx) * decay
        wy = cy + (1.0 - cy) * decay

        dx = ax * wx
        dy = ay * wy

        flow = torch.stack([dx, dy], dim=-1)  # [H,W,2]
        aux = {
            "ax": ax.item(),
            "ay": ay.item(),
            "cx": cx.item(),
            "cy": cy.item(),
            "wx": wx.detach(),
            "wy": wy.detach(),
            "dx": dx.detach(),
            "dy": dy.detach(),
        }
        return flow, aux

    def warp(self, x, flow):
        """
        x:    [B,C,H,W]
        flow: [B,H,W,2] in pixel units
        """
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        base_grid = self.make_base_grid(h, w, device, dtype)
        base_grid = base_grid.unsqueeze(0).expand(b, h, w, 2).contiguous()

        norm_dx = 2.0 * flow[..., 0] / max(w - 1, 1)
        norm_dy = 2.0 * flow[..., 1] / max(h - 1, 1)

        grid = base_grid.clone()
        grid[..., 0] += norm_dx
        grid[..., 1] += norm_dy

        warped = F.grid_sample(
            x, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        return warped

    def forward(self, predictor, gt):
        b, c, h, w = predictor.shape
        device, dtype = predictor.device, predictor.dtype

        all_warped = []
        all_mse = []
        all_flow = []
        all_aux = []

        for k in range(self.num_classes):
            flow_k, aux_k = self.build_flow_from_params(
                self.class_params[k], h, w, device, dtype
            )  # [H,W,2]

            flow_k_b = flow_k.unsqueeze(0).expand(b, h, w, 2)
            warped_k = self.warp(predictor, flow_k_b)

            mse_k = ((warped_k - gt) ** 2).mean(dim=(1, 2, 3))  # [B]

            all_warped.append(warped_k)
            all_mse.append(mse_k)
            all_flow.append(flow_k.detach())
            all_aux.append(aux_k)

        all_warped = torch.stack(all_warped, dim=1)  # [B,9,C,H,W]
        all_mse = torch.stack(all_mse, dim=1)        # [B,9]

        best_idx = all_mse.argmin(dim=1)
        best_mse = all_mse.gather(1, best_idx[:, None]).squeeze(1)

        gather_idx = best_idx.view(b, 1, 1, 1, 1).expand(b, 1, predictor.size(1), predictor.size(2), predictor.size(3))
        best_warped = all_warped.gather(1, gather_idx).squeeze(1)

        return {
            "best_warped": best_warped,
            "best_idx": best_idx,
            "best_mse": best_mse,
            "all_warped": all_warped,
            "all_mse": all_mse,
            "all_flow": all_flow,   # list of [H,W,2]
            "all_aux": all_aux,     # list of dict
        }


def make_demo_predictor_and_gt(h=32, w=32, device="cpu"):
    """
    predictor와 gt를 일부러 살짝 어긋나게 만들어서
    refine이 어떻게 보이는지 확인하기 위한 toy 예시
    """
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij"
    )

    # GT: 경사 + edge + blob 섞은 예시
    gt = (
        0.2
        + 0.3 * (xx / (w - 1))
        + 0.2 * (yy / (h - 1))
        + 0.25 * ((xx + yy) > (0.9 * w)).float()
    )

    # 원형 blob 하나 추가
    cx, cy = 0.35 * w, 0.55 * h
    rr = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
    gt = gt + 0.25 * torch.exp(-(rr ** 2) / (2 * (0.12 * w) ** 2))

    gt = gt.clamp(0.0, 1.0)

    # predictor는 GT를 약간 왼쪽/위로 misalign + blur 비슷하게
    predictor = torch.roll(gt, shifts=(-1, -1), dims=(0, 1))
    predictor = 0.8 * predictor + 0.2 * torch.roll(predictor, shifts=(0, 1), dims=(0, 1))
    predictor = predictor.clamp(0.0, 1.0)

    predictor = predictor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    gt = gt.unsqueeze(0).unsqueeze(0)                # [1,1,H,W]
    return predictor, gt


def visualize_refinement(model, predictor, gt, sample_idx=0, quiver_stride=4):
    out = model(predictor, gt)

    pred_img = predictor[sample_idx, 0].detach().cpu()
    gt_img = gt[sample_idx, 0].detach().cpu()

    best_idx = int(out["best_idx"][sample_idx].item())
    best_mse = float(out["best_mse"][sample_idx].item())

    print(f"Best class index: {best_idx}")
    print(f"Best MSE        : {best_mse:.6f}")
    print()

    for k in range(model.num_classes):
        aux = out["all_aux"][k]
        mse = float(out["all_mse"][sample_idx, k].item())
        print(
            f"class {k}: "
            f"ax={aux['ax']:+.3f}, ay={aux['ay']:+.3f}, "
            f"cx={aux['cx']:.3f}, cy={aux['cy']:.3f}, "
            f"MSE={mse:.6f}"
        )

    # ---- Figure 1: predictor / gt / best ----
    best_img = out["best_warped"][sample_idx, 0].detach().cpu()

    fig1, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(pred_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Predictor")
    axes[1].imshow(gt_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    axes[2].imshow(best_img, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Best refined\nclass={best_idx}")
    axes[3].imshow((best_img - gt_img).abs(), cmap="magma")
    axes[3].set_title("|Best - GT|")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ---- Figure 2: 9개 class 결과 ----
    fig2, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for k in range(model.num_classes):
        warped = out["all_warped"][sample_idx, k, 0].detach().cpu()
        mse = float(out["all_mse"][sample_idx, k].item())
        axes[k].imshow(warped, cmap="gray", vmin=0, vmax=1)
        axes[k].set_title(f"class {k}\nMSE={mse:.5f}")
        axes[k].axis("off")

    plt.tight_layout()
    plt.show()

    # ---- Figure 3: flow 시각화 ----
    fig3, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for k in range(model.num_classes):
        flow = out["all_flow"][k].cpu()  # [H,W,2]
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = torch.sqrt(dx ** 2 + dy ** 2)

        axes[k].imshow(mag, cmap="viridis")
        axes[k].set_title(f"class {k} |flow|")
        axes[k].axis("off")

        h, w = dx.shape
        ys = torch.arange(0, h, quiver_stride)
        xs = torch.arange(0, w, quiver_stride)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        axes[k].quiver(
            xx.numpy(),
            yy.numpy(),
            dx[yy, xx].numpy(),
            dy[yy, xx].numpy(),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
        )

    plt.tight_layout()
    plt.show()

    # ---- Figure 4: dx / dy / wx / wy 를 best class 기준으로 자세히 ----
    best_aux = out["all_aux"][best_idx]
    dx = best_aux["dx"].cpu()
    dy = best_aux["dy"].cpu()
    wx = best_aux["wx"].cpu()
    wy = best_aux["wy"].cpu()

    fig4, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(wx, cmap="viridis")
    axes[0].set_title(f"best class {best_idx}\nwx")
    axes[1].imshow(wy, cmap="viridis")
    axes[1].set_title(f"best class {best_idx}\nwy")
    axes[2].imshow(dx, cmap="coolwarm")
    axes[2].set_title("dx")
    axes[3].imshow(dy, cmap="coolwarm")
    axes[3].set_title("dy")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)

    predictor, gt = make_demo_predictor_and_gt(h=32, w=32, device="cpu")

    model = MergeRefineOracle4Param()

    visualize_refinement(model, predictor, gt, sample_idx=0, quiver_stride=4)

















































import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =========================================================
# 1) Oracle refine model
# =========================================================
class MergeRefineOracle4Param(nn.Module):
    """
    9개 class 각각이 (ax, ay, cx, cy)를 가짐.

    ax, ay : x/y 기본 이동량 (pixel unit)
    cx, cy : center participation (sigmoid -> 0~1)

    최종 flow:
      dx(x,y) = ax * (cx + (1-cx) * decay(x,y))
      dy(x,y) = ay * (cy + (1-cy) * decay(x,y))

    predictor를 9개 방식으로 warp한 뒤,
    gt와의 MSE가 가장 작은 class를 oracle-best로 선택.
    """

    def __init__(self):
        super().__init__()
        self.num_classes = 9

        # [ax, ay, cx_raw, cy_raw]
        init = torch.tensor([
            [ 0.00,  0.00,  8.0,  8.0],   # 0: no-op
            [ 0.25,  0.00, -8.0, -8.0],   # 1: x+
            [-0.25,  0.00, -8.0, -8.0],   # 2: x-
            [ 0.00,  0.25, -8.0, -8.0],   # 3: y+
            [ 0.00, -0.25, -8.0, -8.0],   # 4: y-
            [ 0.25,  0.25, -8.0, -8.0],   # 5: diag ++
            [-0.25, -0.25, -8.0, -8.0],   # 6: diag --
            [ 0.00,  0.25,  0.0, -2.0],   # 7: top-ish
            [ 0.25,  0.00, -2.0,  0.0],   # 8: left-ish
        ], dtype=torch.float32)

        self.class_params = nn.Parameter(init)  # [9,4]

    @staticmethod
    def make_decay_map(h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij"
        )
        # top-left = 1, bottom-right = 0
        decay = 1.0 - 0.5 * (xx + yy)
        return decay.clamp(0.0, 1.0)

    @staticmethod
    def make_base_grid(h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij"
        )
        return torch.stack([xx, yy], dim=-1)  # [H,W,2]

    def build_flow_from_params(self, params, h, w, device, dtype):
        ax, ay, cx_raw, cy_raw = params.unbind(dim=0)

        cx = torch.sigmoid(cx_raw)
        cy = torch.sigmoid(cy_raw)

        decay = self.make_decay_map(h, w, device, dtype)  # [H,W]

        wx = cx + (1.0 - cx) * decay
        wy = cy + (1.0 - cy) * decay

        dx = ax * wx
        dy = ay * wy

        flow = torch.stack([dx, dy], dim=-1)  # [H,W,2]
        aux = {
            "ax": float(ax.detach().cpu()),
            "ay": float(ay.detach().cpu()),
            "cx": float(cx.detach().cpu()),
            "cy": float(cy.detach().cpu()),
            "dx": dx.detach(),
            "dy": dy.detach(),
        }
        return flow, aux

    def warp(self, x, flow):
        """
        x:    [B,C,H,W]
        flow: [B,H,W,2] in pixel units
        """
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        base_grid = self.make_base_grid(h, w, device, dtype)
        base_grid = base_grid.unsqueeze(0).expand(b, h, w, 2).contiguous()

        norm_dx = 2.0 * flow[..., 0] / max(w - 1, 1)
        norm_dy = 2.0 * flow[..., 1] / max(h - 1, 1)

        grid = base_grid.clone()
        grid[..., 0] += norm_dx
        grid[..., 1] += norm_dy

        warped = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        return warped

    def forward(self, predictor, gt):
        """
        predictor, gt: [B,C,H,W]
        """
        b, c, h, w = predictor.shape
        device, dtype = predictor.device, predictor.dtype

        all_warped = []
        all_mse = []
        all_flow = []
        all_aux = []

        for k in range(self.num_classes):
            flow_k, aux_k = self.build_flow_from_params(
                self.class_params[k], h, w, device, dtype
            )  # [H,W,2]

            flow_k_b = flow_k.unsqueeze(0).expand(b, h, w, 2)
            warped_k = self.warp(predictor, flow_k_b)
            mse_k = ((warped_k - gt) ** 2).mean(dim=(1, 2, 3))  # [B]

            all_warped.append(warped_k)
            all_mse.append(mse_k)
            all_flow.append(flow_k.detach())
            all_aux.append(aux_k)

        all_warped = torch.stack(all_warped, dim=1)  # [B,9,C,H,W]
        all_mse = torch.stack(all_mse, dim=1)        # [B,9]

        best_idx = all_mse.argmin(dim=1)             # [B]
        best_mse = all_mse.gather(1, best_idx[:, None]).squeeze(1)

        gather_idx = best_idx.view(b, 1, 1, 1, 1).expand(b, 1, c, h, w)
        best_warped = all_warped.gather(1, gather_idx).squeeze(1)

        return {
            "best_warped": best_warped,
            "best_idx": best_idx,
            "best_mse": best_mse,
            "all_mse": all_mse,
            "all_warped": all_warped,
            "all_flow": all_flow,
            "all_aux": all_aux,
        }


# =========================================================
# 2) Classification network
# =========================================================
class MergeRefineClassifier(nn.Module):
    """
    입력:
      predictor : [B,1,H,W]
      top_map   : [B,1,H,W]
      left_map  : [B,1,H,W]

    출력:
      logits    : [B,9]
    """
    def __init__(self, in_channels=3, num_classes=9, hidden=32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden * 2, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, predictor, top_map, left_map):
        x = torch.cat([predictor, top_map, left_map], dim=1)
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


# =========================================================
# 3) Utility functions
# =========================================================
def build_boundary_maps(top_row, left_col, h, w):
    """
    top_row : [B,1,1,W]
    left_col: [B,1,H,1]

    return:
      top_map  : [B,1,H,W]
      left_map : [B,1,H,W]
    """
    b = top_row.size(0)
    device = top_row.device
    dtype = top_row.dtype

    top_map = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
    left_map = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

    top_map[:, :, 0:1, :] = top_row
    left_map[:, :, :, 0:1] = left_col

    return top_map, left_map


def make_demo_batch(batch_size=64, h=32, w=32, device="cpu"):
    """
    toy predictor / gt / boundary 생성
    gt를 만들고 predictor를 약간 misalign해서 refine이 먹도록 구성
    """
    predictors = []
    gts = []
    top_rows = []
    left_cols = []

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij"
    )

    for i in range(batch_size):
        # 랜덤 구조 만들기
        gx = torch.rand(1, device=device).item() * 0.4 + 0.1
        gy = torch.rand(1, device=device).item() * 0.4 + 0.1
        thr = torch.rand(1, device=device).item() * 0.8 + 0.6
        cx = torch.rand(1, device=device).item() * w
        cy = torch.rand(1, device=device).item() * h
        sigma = torch.rand(1, device=device).item() * 4 + 3

        gt = (
            0.15
            + gx * (xx / (w - 1))
            + gy * (yy / (h - 1))
            + 0.20 * ((xx + yy) > (thr * w)).float()
        )

        rr = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
        gt = gt + 0.25 * torch.exp(-(rr ** 2) / (2 * sigma ** 2))
        gt = gt.clamp(0.0, 1.0)

        # predictor를 GT에서 약간 shifted / blurred
        sx = int(torch.randint(low=-1, high=2, size=(1,), device=device).item())
        sy = int(torch.randint(low=-1, high=2, size=(1,), device=device).item())
        predictor = torch.roll(gt, shifts=(sy, sx), dims=(0, 1))
        predictor = 0.85 * predictor + 0.15 * torch.roll(predictor, shifts=(0, 1), dims=(0, 1))
        predictor = predictor.clamp(0.0, 1.0)

        # boundary는 gt 주변 recon이라고 가정한 toy 값
        top_row = gt[0:1, :].clone().unsqueeze(0).unsqueeze(0)   # [1,1,1,W]
        left_col = gt[:, 0:1].clone().unsqueeze(0).unsqueeze(0)  # [1,1,H,1]

        predictors.append(predictor.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
        gts.append(gt.unsqueeze(0).unsqueeze(0))
        top_rows.append(top_row)
        left_cols.append(left_col)

    predictor = torch.cat(predictors, dim=0)
    gt = torch.cat(gts, dim=0)
    top_row = torch.cat(top_rows, dim=0)
    left_col = torch.cat(left_cols, dim=0)

    return predictor, gt, top_row, left_col


def topk_recall(logits, target, k=3):
    topk = logits.topk(k=k, dim=1).indices
    hit = (topk == target.unsqueeze(1)).any(dim=1).float().mean()
    return float(hit.item())


# =========================================================
# 4) Visualization
# =========================================================
def visualize_oracle_and_classifier(oracle_model, classifier, predictor, gt, top_row, left_col, sample_idx=0):
    with torch.no_grad():
        oracle_out = oracle_model(predictor, gt)

        b, c, h, w = predictor.shape
        top_map, left_map = build_boundary_maps(top_row, left_col, h, w)
        logits = classifier(predictor, top_map, left_map)
        pred_idx = logits.argmax(dim=1)

    pred_img = predictor[sample_idx, 0].cpu()
    gt_img = gt[sample_idx, 0].cpu()
    best_idx = int(oracle_out["best_idx"][sample_idx].item())
    cls_idx = int(pred_idx[sample_idx].item())

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    axes[0, 0].imshow(pred_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Predictor")
    axes[0, 1].imshow(gt_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("GT")

    best_img = oracle_out["best_warped"][sample_idx, 0].cpu()
    axes[0, 2].imshow(best_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Oracle best\nclass={best_idx}")

    cls_img = oracle_out["all_warped"][sample_idx, cls_idx, 0].cpu()
    axes[0, 3].imshow(cls_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title(f"Classifier pred\nclass={cls_idx}")

    axes[0, 4].imshow((best_img - gt_img).abs(), cmap="magma")
    axes[0, 4].set_title("|Oracle - GT|")

    axes[0, 5].imshow((cls_img - gt_img).abs(), cmap="magma")
    axes[0, 5].set_title("|ClsPred - GT|")

    for k in range(6):
        axes[0, k].axis("off")

    for k in range(6):
        class_id = k
        if class_id >= oracle_model.num_classes:
            axes[1, k].axis("off")
            continue
        warped = oracle_out["all_warped"][sample_idx, class_id, 0].cpu()
        mse = float(oracle_out["all_mse"][sample_idx, class_id].item())
        axes[1, k].imshow(warped, cmap="gray", vmin=0, vmax=1)
        axes[1, k].set_title(f"class {class_id}\nMSE={mse:.4f}")
        axes[1, k].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# 5) Training
# =========================================================
def train_classifier(
    epochs=20,
    batch_size=128,
    h=32,
    w=32,
    lr=1e-3,
    device="cpu",
):
    oracle_model = MergeRefineOracle4Param().to(device)
    classifier = MergeRefineClassifier(in_channels=3, num_classes=9, hidden=32).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # 고정 train batch 여러 개 미리 생성
    train_data = [make_demo_batch(batch_size=batch_size, h=h, w=w, device=device) for _ in range(20)]
    val_data = make_demo_batch(batch_size=batch_size, h=h, w=w, device=device)

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0
        total_acc = 0.0
        total_top3 = 0.0

        for predictor, gt, top_row, left_col in train_data:
            with torch.no_grad():
                oracle_out = oracle_model(predictor, gt)
                target_idx = oracle_out["best_idx"]  # [B]

            b, c, hh, ww = predictor.shape
            top_map, left_map = build_boundary_maps(top_row, left_col, hh, ww)

            logits = classifier(predictor, top_map, left_map)
            loss = F.cross_entropy(logits, target_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_idx = logits.argmax(dim=1)
            acc = (pred_idx == target_idx).float().mean().item()
            t3 = topk_recall(logits, target_idx, k=3)

            total_loss += loss.item()
            total_acc += acc
            total_top3 += t3

        total_loss /= len(train_data)
        total_acc /= len(train_data)
        total_top3 /= len(train_data)

        # validation
        classifier.eval()
        with torch.no_grad():
            predictor, gt, top_row, left_col = val_data
            oracle_out = oracle_model(predictor, gt)
            target_idx = oracle_out["best_idx"]

            b, c, hh, ww = predictor.shape
            top_map, left_map = build_boundary_maps(top_row, left_col, hh, ww)

            logits = classifier(predictor, top_map, left_map)
            val_loss = F.cross_entropy(logits, target_idx).item()
            val_acc = (logits.argmax(dim=1) == target_idx).float().mean().item()
            val_top3 = topk_recall(logits, target_idx, k=3)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={total_loss:.4f} "
            f"train_acc={total_acc:.4f} "
            f"train_top3={total_top3:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_top3={val_top3:.4f}"
        )

    return oracle_model, classifier, val_data


# =========================================================
# 6) Main
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    oracle_model, classifier, val_data = train_classifier(
        epochs=20,
        batch_size=128,
        h=32,
        w=32,
        lr=1e-3,
        device=device,
    )

    predictor, gt, top_row, left_col = val_data

    visualize_oracle_and_classifier(
        oracle_model=oracle_model,
        classifier=classifier,
        predictor=predictor,
        gt=gt,
        top_row=top_row,
        left_col=left_col,
        sample_idx=0,
    )








import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MergeRefineOracle4Param(nn.Module):
    """
    각 class는 4개 파라미터를 가짐:
      v_left_raw  : 왼쪽 경계의 vertical displacement raw
      u_top_raw   : 위쪽 경계의 horizontal displacement raw
      decay_x_raw : 왼쪽 경계 motion의 가로 감쇠율 raw
      decay_y_raw : 위쪽 경계 motion의 세로 감쇠율 raw

    최종 flow는:
      dx(x,y) = u_top  * exp(-decay_y * y_norm)
      dy(x,y) = v_left * exp(-decay_x * x_norm)

    여기서
      u_top  = max_shift * tanh(u_top_raw)
      v_left = max_shift * tanh(v_left_raw)

    즉,
      - top boundary에서 horizontal motion이 가장 크고 아래로 갈수록 감소
      - left boundary에서 vertical motion이 가장 크고 오른쪽으로 갈수록 감소
    """

    def __init__(self, max_shift=0.5):
        super().__init__()
        self.num_classes = 8
        self.max_shift = float(max_shift)

        # 각 row = [v_left_raw, u_top_raw, decay_x_raw, decay_y_raw]
        # tanh(raw) * max_shift 가 실제 경계 이동량
        # softplus(raw) 가 실제 decay
        init = torch.tensor([
            [ 0.0,  0.0,  2.0,  2.0],   # 0: no-op
            [ 0.0,  2.0,  2.0,  2.0],   # 1: top -> x+
            [ 0.0, -2.0,  2.0,  2.0],   # 2: top -> x-
            [ 2.0,  0.0,  2.0,  2.0],   # 3: left -> y+
            [-2.0,  0.0,  2.0,  2.0],   # 4: left -> y-
            [ 2.0,  2.0,  2.0,  2.0],   # 5: diag (+x,+y)
            [-2.0, -2.0,  2.0,  2.0],   # 6: diag (-x,-y)
            [ 2.0, -2.0,  2.0,  2.0],   # 7: mixed (-x,+y)
        ], dtype=torch.float32)

        self.class_params = nn.Parameter(init)  # [8,4]

    @staticmethod
    def make_xy_maps(h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij"
        )
        return xx, yy

    @staticmethod
    def make_base_grid(h, w, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij"
        )
        return torch.stack([xx, yy], dim=-1)  # [H,W,2]

    def build_flow_from_params(self, params, h, w, device, dtype):
        v_left_raw, u_top_raw, decay_x_raw, decay_y_raw = params.unbind(dim=0)

        v_left = self.max_shift * torch.tanh(v_left_raw)
        u_top = self.max_shift * torch.tanh(u_top_raw)

        decay_x = F.softplus(decay_x_raw)
        decay_y = F.softplus(decay_y_raw)

        xx, yy = self.make_xy_maps(h, w, device, dtype)

        # top boundary horizontal motion
        wx = torch.exp(-decay_y * yy)   # y=0에서 최대, 아래로 갈수록 감소
        dx = u_top * wx

        # left boundary vertical motion
        wy = torch.exp(-decay_x * xx)   # x=0에서 최대, 오른쪽으로 갈수록 감소
        dy = v_left * wy

        flow = torch.stack([dx, dy], dim=-1)  # [H,W,2]
        aux = {
            "v_left": float(v_left.detach().cpu()),
            "u_top": float(u_top.detach().cpu()),
            "decay_x": float(decay_x.detach().cpu()),
            "decay_y": float(decay_y.detach().cpu()),
            "wx": wx.detach(),
            "wy": wy.detach(),
            "dx": dx.detach(),
            "dy": dy.detach(),
        }
        return flow, aux

    def warp(self, x, flow):
        """
        x:    [B,C,H,W]
        flow: [B,H,W,2] in pixel units
        """
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        base_grid = self.make_base_grid(h, w, device, dtype)
        base_grid = base_grid.unsqueeze(0).expand(b, h, w, 2).contiguous()

        norm_dx = 2.0 * flow[..., 0] / max(w - 1, 1)
        norm_dy = 2.0 * flow[..., 1] / max(h - 1, 1)

        grid = base_grid.clone()
        grid[..., 0] += norm_dx
        grid[..., 1] += norm_dy

        warped = F.grid_sample(
            x, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        return warped

    def forward(self, predictor, gt):
        b, c, h, w = predictor.shape
        device, dtype = predictor.device, predictor.dtype

        all_warped = []
        all_mse = []
        all_flow = []
        all_aux = []

        for k in range(self.num_classes):
            flow_k, aux_k = self.build_flow_from_params(
                self.class_params[k], h, w, device, dtype
            )

            flow_k_b = flow_k.unsqueeze(0).expand(b, h, w, 2)
            warped_k = self.warp(predictor, flow_k_b)

            mse_k = ((warped_k - gt) ** 2).mean(dim=(1, 2, 3))  # [B]

            all_warped.append(warped_k)
            all_mse.append(mse_k)
            all_flow.append(flow_k.detach())
            all_aux.append(aux_k)

        all_warped = torch.stack(all_warped, dim=1)  # [B,8,C,H,W]
        all_mse = torch.stack(all_mse, dim=1)        # [B,8]

        best_idx = all_mse.argmin(dim=1)
        best_mse = all_mse.gather(1, best_idx[:, None]).squeeze(1)

        gather_idx = best_idx.view(b, 1, 1, 1, 1).expand(
            b, 1, predictor.size(1), predictor.size(2), predictor.size(3)
        )
        best_warped = all_warped.gather(1, gather_idx).squeeze(1)

        return {
            "best_warped": best_warped,
            "best_idx": best_idx,
            "best_mse": best_mse,
            "all_warped": all_warped,
            "all_mse": all_mse,
            "all_flow": all_flow,
            "all_aux": all_aux,
        }


def make_demo_predictor_and_gt(h=32, w=32, device="cpu"):
    """
    predictor와 gt를 일부러 살짝 어긋나게 만들어서
    refine이 어떻게 보이는지 확인하기 위한 toy 예시
    """
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij"
    )

    gt = (
        0.2
        + 0.3 * (xx / (w - 1))
        + 0.2 * (yy / (h - 1))
        + 0.25 * ((xx + yy) > (0.9 * w)).float()
    )

    cx, cy = 0.35 * w, 0.55 * h
    rr = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
    gt = gt + 0.25 * torch.exp(-(rr ** 2) / (2 * (0.12 * w) ** 2))
    gt = gt.clamp(0.0, 1.0)

    predictor = torch.roll(gt, shifts=(-1, -1), dims=(0, 1))
    predictor = 0.8 * predictor + 0.2 * torch.roll(predictor, shifts=(0, 1), dims=(0, 1))
    predictor = predictor.clamp(0.0, 1.0)

    predictor = predictor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    gt = gt.unsqueeze(0).unsqueeze(0)                # [1,1,H,W]
    return predictor, gt


def visualize_refinement(model, predictor, gt, sample_idx=0, quiver_stride=4):
    out = model(predictor, gt)

    pred_img = predictor[sample_idx, 0].detach().cpu()
    gt_img = gt[sample_idx, 0].detach().cpu()

    best_idx = int(out["best_idx"][sample_idx].item())
    best_mse = float(out["best_mse"][sample_idx].item())

    print(f"Best class index: {best_idx}")
    print(f"Best MSE        : {best_mse:.6f}")
    print()

    for k in range(model.num_classes):
        aux = out["all_aux"][k]
        mse = float(out["all_mse"][sample_idx, k].item())
        print(
            f"class {k}: "
            f"u_top={aux['u_top']:+.3f}, v_left={aux['v_left']:+.3f}, "
            f"decay_x={aux['decay_x']:.3f}, decay_y={aux['decay_y']:.3f}, "
            f"MSE={mse:.6f}"
        )

    # ---- Figure 1: predictor / gt / best ----
    best_img = out["best_warped"][sample_idx, 0].detach().cpu()

    fig1, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(pred_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Predictor")
    axes[1].imshow(gt_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    axes[2].imshow(best_img, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Best refined\nclass={best_idx}")
    axes[3].imshow((best_img - gt_img).abs(), cmap="magma")
    axes[3].set_title("|Best - GT|")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # ---- Figure 2: class 결과 ----
    n = model.num_classes
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig2, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for k in range(model.num_classes):
        warped = out["all_warped"][sample_idx, k, 0].detach().cpu()
        mse = float(out["all_mse"][sample_idx, k].item())
        axes[k].imshow(warped, cmap="gray", vmin=0, vmax=1)
        axes[k].set_title(f"class {k}\nMSE={mse:.5f}")
        axes[k].axis("off")

    for k in range(model.num_classes, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.show()

    # ---- Figure 3: flow 시각화 ----
    fig3, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for k in range(model.num_classes):
        flow = out["all_flow"][k].cpu()
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = torch.sqrt(dx ** 2 + dy ** 2)

        axes[k].imshow(mag, cmap="viridis")
        axes[k].set_title(f"class {k} |flow|")
        axes[k].axis("off")

        h, w = dx.shape
        ys = torch.arange(0, h, quiver_stride)
        xs = torch.arange(0, w, quiver_stride)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        axes[k].quiver(
            xx.numpy(),
            yy.numpy(),
            dx[yy, xx].numpy(),
            dy[yy, xx].numpy(),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
        )

    for k in range(model.num_classes, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.show()

    # ---- Figure 4: dx / dy / wx / wy 를 best class 기준으로 자세히 ----
    best_aux = out["all_aux"][best_idx]
    dx = best_aux["dx"].cpu()
    dy = best_aux["dy"].cpu()
    wx = best_aux["wx"].cpu()
    wy = best_aux["wy"].cpu()

    fig4, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(wx, cmap="viridis")
    axes[0].set_title(f"best class {best_idx}\nwx (top->down decay)")
    axes[1].imshow(wy, cmap="viridis")
    axes[1].set_title(f"best class {best_idx}\nwy (left->right decay)")
    axes[2].imshow(dx, cmap="coolwarm")
    axes[2].set_title("dx")
    axes[3].imshow(dy, cmap="coolwarm")
    axes[3].set_title("dy")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)

    predictor, gt = make_demo_predictor_and_gt(h=32, w=32, device="cpu")

    model = MergeRefineOracle4Param(max_shift=0.5)

    visualize_refinement(model, predictor, gt, sample_idx=0, quiver_stride=4)




