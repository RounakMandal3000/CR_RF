import os
import numpy as np
import rasterio
from tqdm import tqdm
from s2cloudless import S2PixelCloudDetector
import multiprocessing
import argparse


# ------------------------------------------------
# Worker-level initializer (runs once per process)
# Creates a process-local cloud_detector to avoid pickling/sharing issues
# ------------------------------------------------
def init_worker(threshold=0.4, average_over=4, dilation_size=2, all_bands=True):
    global cloud_detector
    cloud_detector = S2PixelCloudDetector(
        threshold=threshold,
        average_over=average_over,
        dilation_size=dilation_size,
        all_bands=all_bands,
    )


# ------------------------------------------------
# Function to load a single Sentinel-2 image
# ------------------------------------------------
def load_s2_image(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (C, H, W)
    return np.transpose(arr, (1, 2, 0)) / 10000.0


# ------------------------------------------------
# Function to process a single file (top-level function for multiprocessing)
# Accepts a single tuple argument to make it pickle-friendly on Windows
# ------------------------------------------------
def process_single(args):
    # args: (full_path, rel_path, input_folder, output_folder, force)
    full_path, rel_path, input_folder, output_folder, force = args
    try:
        rel_dir = os.path.dirname(rel_path)
        out_dir = os.path.join(output_folder, rel_dir)
        fname = os.path.basename(rel_path)
        out_name = fname.replace(".tif", "_cloudprob.npy")
        out_path = os.path.join(out_dir, out_name)

        # If output already exists and not forced, skip processing
        if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return (full_path, "skipped")

        # ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

        s2 = load_s2_image(full_path)
        # Use the process-local detector created by init_worker
        cloud_prob = cloud_detector.get_cloud_probability_maps(np.expand_dims(s2, axis=0))[0]

        # write output
        np.save(out_path, cloud_prob)
        return (full_path, None)
    except Exception as e:
        return (full_path, str(e))


# ------------------------------------------------
# Process folder in parallel using multiprocessing.Pool
# ------------------------------------------------
def process_folder_parallel(input_folder, output_folder, num_workers=None, detector_kwargs=None, force=False):
    if detector_kwargs is None:
        detector_kwargs = {}

    os.makedirs(output_folder, exist_ok=True)

    # Collect all TIFF files with their relative paths
    tif_paths = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(".tif"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_folder)
                tif_paths.append((full_path, rel_path))

    total = len(tif_paths)
    print(f"Found {total} TIFF files.\n")

    if total == 0:
        return

    # Build iterable of args for worker (include force flag)
    arg_iter = ((p, rel, input_folder, output_folder, force) for (p, rel) in tif_paths)

    # On Windows the default start method is 'spawn' which requires the
    # if __name__ == '__main__' guard (handled below). We pass the init args
    # so each worker creates its own detector instance (no shared state).
    workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting pool with {workers} worker(s)...")

    errors = []
    skipped_count = 0
    processed_count = 0
    initargs = tuple(detector_kwargs.get(k) for k in ('threshold', 'average_over', 'dilation_size', 'all_bands')) if detector_kwargs else ()
    with multiprocessing.Pool(processes=workers, initializer=init_worker, initargs=initargs) as pool:
        # Using imap_unordered keeps memory use low and allows progress bar
        for result in tqdm(pool.imap_unordered(process_single, arg_iter), total=total, desc="Processing Sentinel-2 tiles", unit="img"):
            full_path, err = result
            if err == "skipped":
                skipped_count += 1
            elif err:
                errors.append((full_path, err))
            else:
                processed_count += 1

    print(f"\nSummary: processed={processed_count}, skipped={skipped_count}, errors={len(errors)}")
    if errors:
        print(f"\n⚠️ Finished with {len(errors)} errors. First few:")
        for fp, e in errors[:10]:
            print(f" - {fp}: {e}")
    else:
        print(f"\n✅ Completed! All outputs stored under: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel cloud probability extractor for Sentinel-2 tiles")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing .tif files")
    parser.add_argument("--output", "-o", required=True, help="Output folder to store .npy cloud probability maps")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of worker processes (default: CPU-1)")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--average_over", type=int, default=4)
    parser.add_argument("--dilation_size", type=int, default=2)
    parser.add_argument("--all_bands", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)

    args = parser.parse_args()

    detector_kwargs = {
        'threshold': args.threshold,
        'average_over': args.average_over,
        'dilation_size': args.dilation_size,
        'all_bands': args.all_bands,
    }

    # Multiprocessing on Windows requires the entry point guard (we are inside it)
    process_folder_parallel(args.input, args.output, num_workers=args.workers, detector_kwargs=detector_kwargs)
