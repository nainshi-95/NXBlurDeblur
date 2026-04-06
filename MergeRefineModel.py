import torch
import torch.nn as nn
import torch.nn.functional as F


class MergeRefineOracle(nn.Module):
    """
    predictor를 9개 predefined transform으로 모두 변형한 뒤,
    GT와의 MSE가 가장 작은 결과만 선택해서 반환한다.

    Input
    -----
    predictor : [B, C, H, W]
    gt        : [B, C, H, W]

    Return
    ------
    best_warped : [B, C, H, W]
    best_idx    : [B]          # 0~8
    best_mse    : [B]
    all_mse     : [B, 9]
    all_warped  : [B, 9, C, H, W]
    """
    def __init__(self, shift_amount: float = 0.25):
        super().__init__()
        self.shift_amount = shift_amount
        self.num_classes = 9

    @staticmethod
    def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        return grid

    @staticmethod
    def _make_decay_maps(h: int, w: int, device: torch.device, dtype: torch.dtype):
        """
        공통 decay:
          top-left에서 가장 크고 bottom-right로 갈수록 작아짐

        top-dominant:
          top에서 크고 아래로 갈수록 작아짐

        left-dominant:
          left에서 크고 오른쪽으로 갈수록 작아짐
        """
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )

        # top-left -> bottom-right decay
        common = 1.0 - 0.5 * (xx + yy)
        common = common.clamp(min=0.0, max=1.0)

        # top dominant
        top = 1.0 - yy
        top = top.clamp(min=0.0, max=1.0)

        # left dominant
        left = 1.0 - xx
        left = left.clamp(min=0.0, max=1.0)

        return common, top, left

    def _make_flow_templates(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        """
        9개 클래스의 flow template 생성.
        반환 shape: [9, H, W, 2]
        """
        a = self.shift_amount
        common, top_dom, left_dom = self._make_decay_maps(h, w, device, dtype)

        zeros = torch.zeros_like(common)

        # 각 class의 dx, dy
        flows = []

        # 0: no-op
        flows.append(torch.stack([zeros, zeros], dim=-1))

        # 1: x+
        flows.append(torch.stack([+a * common, zeros], dim=-1))

        # 2: x-
        flows.append(torch.stack([-a * common, zeros], dim=-1))

        # 3: y+
        flows.append(torch.stack([zeros, +a * common], dim=-1))

        # 4: y-
        flows.append(torch.stack([zeros, -a * common], dim=-1))

        # 5: diag ++
        flows.append(torch.stack([+a * common, +a * common], dim=-1))

        # 6: diag --
        flows.append(torch.stack([-a * common, -a * common], dim=-1))

        # 7: top-dominant (vertical correction)
        flows.append(torch.stack([zeros, +a * top_dom], dim=-1))

        # 8: left-dominant (horizontal correction)
        flows.append(torch.stack([+a * left_dom, zeros], dim=-1))

        flows = torch.stack(flows, dim=0)  # [9, H, W, 2]
        return flows

    def _warp_with_flow(self, x: torch.Tensor, flow_px: torch.Tensor):
        """
        x       : [B, C, H, W]
        flow_px : [B, H, W, 2]  (pixel unit, dx/dy)

        return  : [B, C, H, W]
        """
        b, c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        base_grid = self._make_base_grid(h, w, device, dtype)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(b, h, w, 2).contiguous()

        # pixel flow -> normalized coords
        norm_dx = 2.0 * flow_px[..., 0] / max(w - 1, 1)
        norm_dy = 2.0 * flow_px[..., 1] / max(h - 1, 1)

        sample_grid = base_grid.clone()
        sample_grid[..., 0] = sample_grid[..., 0] + norm_dx
        sample_grid[..., 1] = sample_grid[..., 1] + norm_dy

        warped = F.grid_sample(
            x,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped

    def forward(self, predictor: torch.Tensor, gt: torch.Tensor):
        """
        predictor: [B, C, H, W]
        gt       : [B, C, H, W]
        """
        assert predictor.shape == gt.shape, "predictor와 gt shape가 같아야 함"

        b, c, h, w = predictor.shape
        device = predictor.device
        dtype = predictor.dtype

        flow_templates = self._make_flow_templates(h, w, device, dtype)  # [9,H,W,2]

        warped_list = []
        mse_list = []

        for k in range(self.num_classes):
            flow_k = flow_templates[k].unsqueeze(0).expand(b, h, w, 2)  # [B,H,W,2]
            warped_k = self._warp_with_flow(predictor, flow_k)           # [B,C,H,W]

            mse_k = ((warped_k - gt) ** 2).mean(dim=(1, 2, 3))           # [B]

            warped_list.append(warped_k)
            mse_list.append(mse_k)

        all_warped = torch.stack(warped_list, dim=1)  # [B,9,C,H,W]
        all_mse = torch.stack(mse_list, dim=1)        # [B,9]

        best_idx = all_mse.argmin(dim=1)              # [B]
        best_mse = all_mse.gather(1, best_idx.unsqueeze(1)).squeeze(1)  # [B]

        # best warped gather
        gather_idx = best_idx.view(b, 1, 1, 1, 1).expand(b, 1, c, h, w)
        best_warped = all_warped.gather(dim=1, index=gather_idx).squeeze(1)  # [B,C,H,W]

        return {
            "best_warped": best_warped,
            "best_idx": best_idx,
            "best_mse": best_mse,
            "all_mse": all_mse,
            "all_warped": all_warped,
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 4, 1, 16, 16
    predictor = torch.rand(B, C, H, W)
    gt = torch.rand(B, C, H, W)

    model = MergeRefineOracle(shift_amount=0.25)
    out = model(predictor, gt)

    print("best_warped:", out["best_warped"].shape)  # [B,C,H,W]
    print("best_idx:", out["best_idx"])              # [B]
    print("best_mse:", out["best_mse"])              # [B]
    print("all_mse:", out["all_mse"].shape)          # [B,9]
    print("all_warped:", out["all_warped"].shape)    # [B,9,C,H,W]
