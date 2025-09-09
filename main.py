# app.py
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image, ImageSequence, ImageChops

# PaddleOCR + GPU checks
import paddle
from paddleocr import PaddleOCR  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# GPU enforcement
# ──────────────────────────────────────────────────────────────────────────────
def assert_gpu_available() -> None:
    if not paddle.device.is_compiled_with_cuda():
        sys.exit("ERROR: PaddlePaddle is not compiled with CUDA. Install the GPU build.")
    try:
        paddle.set_device("gpu")
    except Exception as e:
        sys.exit(f"ERROR: Failed to set device to GPU: {e}")
    try:
        # Best-effort device count check
        from paddle.device import cuda as pd_cuda  # type: ignore
        if hasattr(pd_cuda, "device_count") and pd_cuda.device_count() < 1:
            sys.exit("ERROR: No CUDA GPUs detected.")
    except Exception:
        # If this import isn't present on your version, set_device('gpu') above is still authoritative.
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Minimal image helpers
# ──────────────────────────────────────────────────────────────────────────────
def _img_to_bgr_array(img: Image.Image) -> np.ndarray:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _preprocess_for_ocr(img: Image.Image) -> np.ndarray:
    # Light upscale for very small scans
    max_dim = max(img.size)
    if max_dim < 1500:
        scale = min(2000 / max_dim, 2.0)
        if scale > 1.01:
            nw, nh = int(img.width * scale), int(img.height * scale)
            img = img.resize((nw, nh), Image.LANCZOS)

    bgr = _img_to_bgr_array(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 25:  # low-contrast heuristic
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15
        )
        bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return bgr


# ──────────────────────────────────────────────────────────────────────────────
# OCR core
# ──────────────────────────────────────────────────────────────────────────────
def warmup(ocr: PaddleOCR) -> None:
    # Tiny white image — ensures model weights are loaded on GPU
    img = Image.new("RGB", (32, 32), "white")
    bgr = _img_to_bgr_array(img)
    _ = ocr.predict(bgr)

def parse_ocr_result_count(result) -> Tuple[int, int]:
    """
    Return (lines_count, boxes_count) from PaddleOCR .ocr() result.
    result ≈ [ [ [box, (text, score)], ... ] ]
    """
    lines = 0
    boxes = 0
    if not result:
        return 0, 0
    page_items = result[0] if isinstance(result[0], list) else result
    for entry in page_items:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            boxes += 1
            meta = entry[1]
            if isinstance(meta, (list, tuple)) and len(meta) >= 1:
                if str(meta[0]).strip():
                    lines += 1
    return lines, boxes

def ocr_tiff_file(fp: Path, ocr: PaddleOCR) -> Tuple[int, int]:
    """
    Returns (pages_processed, total_lines_detected). Any exception should bubble up.
    """
    pages = 0
    total_lines = 0
    with Image.open(fp) as im:
        for _i, frame in enumerate(ImageSequence.Iterator(im), start=1):
            pages += 1
            # Some multi-page TIFFs repeat identical blank frames; cheap skip:
            try:
                if frame.getbbox() is None:  # fully blank
                    continue
            except Exception:
                pass
            bgr = _preprocess_for_ocr(frame)
            result = ocr.predict(bgr)
            lines, _ = parse_ocr_result_count(result)
            total_lines += lines
    return pages, total_lines


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner (sequential)
# ──────────────────────────────────────────────────────────────────────────────
def run_batch(directory: Path) -> None:
    if not directory.is_dir():
        sys.exit(f"ERROR: Not a directory: {directory}")

    files: List[Path] = sorted(
        [*directory.glob("*.tif"), *directory.glob("*.tiff")],
        key=lambda p: p.name.lower(),
    )
    if not files:
        print("No .tif/.tiff files found.")
        return

    # Enforce GPU
    assert_gpu_available()

    # Init OCR on GPU
    ocr = PaddleOCR(lang="en")
    warmup(ocr)

    start = time.perf_counter()
    ok = 0
    failed = 0
    total_pages = 0
    total_lines = 0
    errors: List[Tuple[str, str]] = []

    n = len(files)
    print(f"Discovered {n} TIFF(s). Starting sequential OCR on GPU…")

    for i, fp in enumerate(files, start=1):
        t0 = time.perf_counter()
        try:
            pages, lines = ocr_tiff_file(fp, ocr)
            dt = time.perf_counter() - t0
            ok += 1
            total_pages += pages
            total_lines += lines
            print(f"[{i}/{n}] OK   {fp.name:40s} pages={pages:<4d} lines={lines:<5d} {dt:6.2f}s")
        except Exception as e:
            dt = time.perf_counter() - t0
            failed += 1
            msg = f"{type(e).__name__}: {e}"
            errors.append((fp.name, msg))
            print(f"[{i}/{n}] FAIL {fp.name:40s} {dt:6.2f}s  -> {msg}")

    elapsed = time.perf_counter() - start

    # Summary
    print("\n── Batch Summary ─────────────────────────────────────────")
    print(f"Directory:      {directory}")
    print(f"Discovered:     {n}")
    print(f"Processed OK:   {ok}")
    print(f"Failed:         {failed}")
    print(f"Pages (total):  {total_pages}")
    print(f"Lines (total):  {total_lines}")
    print(f"Elapsed:        {elapsed:.2f}s")
    if errors:
        print("\nErrors:")
        for name, emsg in errors:
            print(f"  - {name}: {emsg}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Sequentially OCR all .tif/.tiff files in a directory on GPU (required)."
    )
    ap.add_argument("directory", type=str, help="Path to directory containing .tif/.tiff files.")
    args = ap.parse_args()

    run_batch(Path(args.directory).expanduser().resolve())


if __name__ == "__main__":
    main()
