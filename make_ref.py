#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MetadataRow:
    poc: int
    x: int
    y: int
    h: int
    w: int
    inter_dir: int
    ref_list: int
    ref_idx: int
    ref_poc: int
    mv_hor: int
    mv_ver: int


@dataclass
class FolderSummary:
    processed_folders: int = 0
    processed_txt: int = 0
    total_blocks: int = 0
    skipped_rows: int = 0
    failed_txt: int = 0

    def merge(self, other: "FolderSummary") -> None:
        self.processed_folders += other.processed_folders
        self.processed_txt += other.processed_txt
        self.total_blocks += other.total_blocks
        self.skipped_rows += other.skipped_rows
        self.failed_txt += other.failed_txt


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


class ReconYuvReader:
    def __init__(self, yuv_path: Path, width: int, height: int, bit_depth: int = 10) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame size: {width}x{height}")
        if bit_depth != 10:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        self.yuv_path = yuv_path
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.dtype = np.uint16
        self.bytes_per_sample = 2
        self.y_plane_bytes = width * height * self.bytes_per_sample
        self.u_plane_bytes = (width // 2) * (height // 2) * self.bytes_per_sample
        self.v_plane_bytes = self.u_plane_bytes
        self.total_frame_bytes = self.y_plane_bytes + self.u_plane_bytes + self.v_plane_bytes
        self.file_size = self.yuv_path.stat().st_size
        self.total_frames = self.file_size // self.total_frame_bytes
        self._fp: BinaryIO | None = None
        self._cached_poc: int | None = None
        self._cached_y_plane: np.ndarray | None = None

        if self.total_frame_bytes <= 0:
            raise ValueError("Computed total frame size must be positive")
        if self.file_size < self.total_frame_bytes:
            raise ValueError(f"Recon file is too small for one frame: {yuv_path}")

    def __enter__(self) -> "ReconYuvReader":
        self._fp = self.yuv_path.open("rb")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._fp is not None:
            self._fp.close()
        self._fp = None
        self._cached_poc = None
        self._cached_y_plane = None

    def get_y_plane(self, poc: int) -> np.ndarray:
        if poc < 0:
            raise ValueError(f"POC must be non-negative: {poc}")
        if poc >= self.total_frames:
            raise ValueError(f"POC {poc} is out of range for total_frames={self.total_frames}")
        if self._fp is None:
            raise RuntimeError("ReconYuvReader is not open")
        if self._cached_poc == poc and self._cached_y_plane is not None:
            return self._cached_y_plane

        offset = poc * self.total_frame_bytes
        self._fp.seek(offset)
        y_plane = np.fromfile(self._fp, dtype=self.dtype, count=self.width * self.height)
        if y_plane.size != self.width * self.height:
            raise ValueError(
                f"Failed to read Y plane for ref_poc={poc} from {self.yuv_path}: "
                f"expected {self.width * self.height} samples, got {y_plane.size}"
            )

        frame = y_plane.reshape((self.height, self.width))
        self._cached_poc = poc
        self._cached_y_plane = frame
        return frame


def warn(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr, flush=True)


def validate_dir(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"{name} is not a directory: {resolved}")
    return resolved


def parse_resolution_from_name(name: str) -> tuple[int, int]:
    match = re.search(r"(\d+)x(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse resolution from folder name: {name}")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid parsed resolution from folder name: {name}")
    return width, height


def find_recon_file(recon_dir: Path, subfolder_name: str) -> Path:
    return recon_dir / f"{subfolder_name}.yuv"


def parse_metadata_line(line: str, txt_path: Path, line_number: int) -> MetadataRow | None:
    stripped = line.strip()
    if not stripped:
        return None

    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) < 11:
        return None

    try:
        row = MetadataRow(
            poc=int(parts[0]),
            x=int(parts[1]),
            y=int(parts[2]),
            h=int(parts[3]),
            w=int(parts[4]),
            inter_dir=int(parts[5]),
            ref_list=int(parts[6]),
            ref_idx=int(parts[7]),
            ref_poc=int(parts[8]),
            mv_hor=int(parts[9]),
            mv_ver=int(parts[10]),
        )
    except ValueError:
        return None

    if row.ref_poc < 0 or row.x < 0 or row.y < 0 or row.h <= 0 or row.w <= 0:
        return None
    return row


def y_plane_to_tensor(y_plane: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(y_plane.astype(np.float32, copy=False))


def extract_reference_block(
    y_plane: np.ndarray,
    row: MetadataRow,
    interpolator: Simple12TapInterp,
) -> np.ndarray:
    ref_tensor = y_plane_to_tensor(y_plane)
    block = interpolator.extract_block(
        reference=ref_tensor,
        x=row.x - 8,
        y=row.y - 8,
        h=row.h + 8,
        w=row.w + 8,
        mv_x=row.mv_hor,
        mv_y=row.mv_ver,
        out_dtype=torch.int32,
    )
    return block.to(dtype=torch.uint16).cpu().numpy().astype("<u2", copy=False)


def process_metadata_file(
    txt_path: Path,
    output_path: Path,
    reader: ReconYuvReader,
    interpolator: Simple12TapInterp,
) -> tuple[int, int]:
    block_count = 0
    skipped_rows = 0

    with txt_path.open("r", encoding="utf-8") as metadata_fp, output_path.open("wb") as output_fp:
        for line_number, line in enumerate(metadata_fp, start=1):
            row = parse_metadata_line(line, txt_path, line_number)
            if row is None:
                skipped_rows += 1
                continue

            if row.ref_poc >= reader.total_frames:
                skipped_rows += 1
                continue

            try:
                y_plane = reader.get_y_plane(row.ref_poc)
                block = extract_reference_block(y_plane, row, interpolator)
                block.tofile(output_fp)
                block_count += 1
            except Exception:
                skipped_rows += 1

    return block_count, skipped_rows


def process_subfolder(
    subfolder: Path,
    recon_dir: Path,
    overwrite: bool,
) -> FolderSummary:
    summary = FolderSummary()
    metadata_dir = subfolder / "metadata"
    if not metadata_dir.is_dir():
        return summary

    recon_yuv_path = find_recon_file(recon_dir, subfolder.name)
    if not recon_yuv_path.is_file():
        return summary

    txt_files = sorted(metadata_dir.glob("*.txt"))
    if not txt_files:
        return summary

    width, height = parse_resolution_from_name(subfolder.name)
    interpolator = Simple12TapInterp(bit_depth=10)
    ref_output_dir = subfolder / "ref"
    ref_output_dir.mkdir(parents=True, exist_ok=True)

    with ReconYuvReader(recon_yuv_path, width, height, bit_depth=10) as reader:
        summary.processed_folders = 1
        for txt_path in txt_files:
            output_path = ref_output_dir / f"{txt_path.stem}.bin"
            if output_path.exists() and not overwrite:
                continue

            try:
                block_count, skipped_rows = process_metadata_file(
                    txt_path=txt_path,
                    output_path=output_path,
                    reader=reader,
                    interpolator=interpolator,
                )
                summary.processed_txt += 1
                summary.total_blocks += block_count
                summary.skipped_rows += skipped_rows
            except Exception:
                summary.failed_txt += 1
                try:
                    if output_path.exists():
                        output_path.unlink()
                except OSError:
                    pass

    return summary


def worker_process_subfolder(
    subfolder_str: str,
    recon_dir_str: str,
    overwrite: bool,
) -> FolderSummary:
    torch.set_num_threads(1)
    subfolder = Path(subfolder_str)
    recon_dir = Path(recon_dir_str)
    return process_subfolder(subfolder=subfolder, recon_dir=recon_dir, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reference Y-plane blocks from metadata using subfolder-level multiprocessing."
    )
    parser.add_argument("--data_dump_root", type=Path, default=Path("./data_dump"))
    parser.add_argument("--recon_dir", type=Path, default=Path("./test_recon"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--folder_pattern", type=str, default="*")
    parser.add_argument("--max_workers", type=int, default=None)
    return parser.parse_args()


def collect_subfolders(data_dump_root: Path, folder_pattern: str) -> list[Path]:
    return sorted(
        path
        for path in data_dump_root.iterdir()
        if path.is_dir() and fnmatch.fnmatch(path.name, folder_pattern)
    )


def resolve_max_workers(max_workers: int | None, task_count: int) -> int:
    if max_workers is not None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        return min(max_workers, task_count)

    cpu_count = os.cpu_count() or 1
    return min(cpu_count, task_count)


def print_summary(summary: FolderSummary) -> None:
    print(f"processed folder count : {summary.processed_folders}")
    print(f"processed txt count    : {summary.processed_txt}")
    print(f"total block count      : {summary.total_blocks}")
    print(f"skipped row count      : {summary.skipped_rows}")
    print(f"failed txt count       : {summary.failed_txt}")


def main() -> int:
    args = parse_args()

    try:
        data_dump_root = validate_dir(args.data_dump_root, "data_dump_root")
        recon_dir = validate_dir(args.recon_dir, "recon_dir")
        subfolders = collect_subfolders(data_dump_root, args.folder_pattern)
        if not subfolders:
            raise ValueError(
                f"no matching subfolders found under {data_dump_root} for pattern {args.folder_pattern!r}"
            )
        max_workers = resolve_max_workers(args.max_workers, len(subfolders))
    except (FileNotFoundError, ValueError) as exc:
        warn(str(exc))
        return 1

    summary = FolderSummary()

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    worker_process_subfolder,
                    str(subfolder),
                    str(recon_dir),
                    args.overwrite,
                ): subfolder
                for subfolder in subfolders
            }

            for future in as_completed(futures):
                subfolder = futures[future]
                try:
                    folder_summary = future.result()
                    summary.merge(folder_summary)
                except Exception as exc:
                    warn(f"failed to process folder {subfolder}: {exc}")
    except Exception as exc:
        warn(str(exc))
        return 1

    print_summary(summary)
    return 0 if summary.failed_txt == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
