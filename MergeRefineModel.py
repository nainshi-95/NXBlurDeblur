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
