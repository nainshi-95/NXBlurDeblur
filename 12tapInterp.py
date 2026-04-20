class Simple12TapInterp(nn.Module):
    FILTERS = torch.tensor(
        [
            [0, 0, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0],
            [-1, 2, -3, 6, -14, 254, 16, -7, 4, -2, 1, 0],
            [-1, 3, -7, 12, -26, 249, 35, -15, 8, -4, 2, 0],
            [-2, 5, -9, 17, -36, 241, 54, -22, 12, -6, 3, -1],
            [-2, 5, -11, 21, -43, 230, 75, -29, 15, -8, 4, -1],
            [-2, 6, -13, 24, -48, 216, 97, -36, 19, -10, 4, -1],
            [-2, 7, -14, 25, -51, 200, 119, -42, 22, -12, 5, -1],
            [-2, 7, -14, 26, -51, 181, 140, -46, 24, -13, 6, -2],
            [-2, 6, -13, 25, -50, 162, 162, -50, 25, -13, 6, -2],
            [-2, 6, -13, 24, -46, 140, 181, -51, 26, -14, 7, -2],
            [-1, 5, -12, 22, -42, 119, 200, -51, 25, -14, 7, -2],
            [-1, 4, -10, 19, -36, 97, 216, -48, 24, -13, 6, -2],
            [-1, 4, -8, 15, -29, 75, 230, -43, 21, -11, 5, -2],
            [-1, 3, -6, 12, -22, 54, 241, -36, 17, -9, 5, -2],
            [0, 2, -4, 8, -15, 35, 249, -26, 12, -7, 3, -1],
            [0, 1, -2, 4, -7, 16, 254, -14, 6, -3, 2, -1],
        ],
        dtype=torch.float32,
    ) / 256.0

    def __init__(self, bit_depth: int = 10) -> None:
        super().__init__()
        self.bit_depth = bit_depth
        self.register_buffer("filters", self.FILTERS.clone())

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x[None, None].to(dtype=torch.float32)
        if x.ndim == 4 and x.shape[:2] == (1, 1):
            return x.to(dtype=torch.float32)
        raise ValueError(f"Expected [H,W] or [1,1,H,W], got {tuple(x.shape)}")

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0, (1 << self.bit_depth) - 1)

    def _get_kernel_x(self, frac_x: int) -> torch.Tensor:
        return self.filters[frac_x].view(1, 1, 1, 12)

    def _get_kernel_y(self, frac_y: int) -> torch.Tensor:
        return self.filters[frac_y].view(1, 1, 12, 1)

    @staticmethod
    def _split_mv(mv: int) -> tuple[int, int]:
        return mv // 16, mv & 15

    def _extract_patch(self, ref: torch.Tensor, x: int, y: int, w: int, h: int) -> torch.Tensor:
        _, _, frame_h, frame_w = ref.shape
        pad_left = max(0, -x)
        pad_top = max(0, -y)
        pad_right = max(0, x + w - frame_w)
        pad_bottom = max(0, y + h - frame_h)
        ref = F.pad(ref, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        x0 = x + pad_left
        y0 = y + pad_top
        return ref[:, :, y0:y0 + h, x0:x0 + w]

    def extract_block(
        self,
        reference: torch.Tensor,
        x: int,
        y: int,
        h: int,
        w: int,
        mv_x: int,
        mv_y: int,
        out_dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        ref = self._to_4d(reference)
        int_x, frac_x = self._split_mv(mv_x)
        int_y, frac_y = self._split_mv(mv_y)
        patch = self._extract_patch(ref, x + int_x - 5, y + int_y - 5, w + 11, h + 11)

        if frac_x != 0:
            patch = F.conv2d(patch, self._get_kernel_x(frac_x))
        if frac_y != 0:
            patch = F.conv2d(patch, self._get_kernel_y(frac_y))

        if frac_x == 0 and frac_y == 0:
            out = patch[:, :, 5:5 + h, 5:5 + w]
        elif frac_x != 0 and frac_y == 0:
            out = patch[:, :, 5:5 + h, :]
        elif frac_x == 0 and frac_y != 0:
            out = patch[:, :, :, 5:5 + w]
        else:
            out = patch

        out = self._clip(out[0, 0])
        if out_dtype.is_floating_point:
            return out.to(out_dtype)
        return torch.round(out).to(out_dtype)
