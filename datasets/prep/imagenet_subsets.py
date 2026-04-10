"""Build slipstream caches for ImageNet-100 and ImageNet-10 subsets.

Reads the full ImageNet-1K cache, filters to the subset classes, adds
multi-label fields, and writes a new standalone cache.

Usage::

    # Build ImageNet-100 val (JPEG)
    python -m datasets.prep.imagenet_subsets --subset in100 --split val --fmt jpeg

    # Build ImageNet-10 train (JPEG)
    python -m datasets.prep.imagenet_subsets --subset in10 --split train --fmt jpeg

    # Build all subsets and splits
    python -m datasets.prep.imagenet_subsets --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .._configs.imagenet_subsets import (
    IN100_TO_IN1000,
    IN10_TO_IN1000,
    IN10_TO_IN100,
    IN1000_TO_IN100,
    IN1000_TO_IN10,
)


# ---------------------------------------------------------------------------
# Subset definitions
# ---------------------------------------------------------------------------

SUBSET_DEFS = {
    "in100": {
        "name": "imagenet100",
        "num_classes": 100,
        "in1000_classes": set(IN100_TO_IN1000.values()),
        "label_fields": {
            "label": IN1000_TO_IN100,       # native subset label
            "in100_label": IN1000_TO_IN100,  # explicit in100 label
        },
    },
    "in10": {
        "name": "imagenet10",
        "num_classes": 10,
        "in1000_classes": set(IN10_TO_IN1000.values()),
        "label_fields": {
            "label": IN1000_TO_IN10,         # native subset label
            "in10_label": IN1000_TO_IN10,    # explicit in10 label
            "in100_label": IN1000_TO_IN100,  # in100 label (in10 ⊂ in100)
        },
    },
}

# Source cache naming pattern (must match imagenet1k config)
SOURCE_CACHE_PATTERN = "imagenet1k-s256_l512-{fmt}-{split}"


def _resolve_source_cache_dir(split: str, fmt: str, cache_base: Path | None = None) -> Path:
    """Resolve the local path to the imagenet1k source cache."""
    if cache_base is None:
        from ..runtime_platform import configure_slipstream_cache
        cache_base = Path(configure_slipstream_cache())

    cache_name = SOURCE_CACHE_PATTERN.format(fmt=fmt, split=split)
    return cache_base / cache_name


def build_subset_cache(
    subset: str,
    split: str = "val",
    fmt: str = "jpeg",
    cache_base: Path | None = None,
    output_base: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Build a subset cache from the ImageNet-1K source cache.

    Args:
        subset: Subset name ("in100" or "in10").
        split: Dataset split ("val" or "train").
        fmt: Image format ("jpeg" or "yuv420").
        cache_base: Base directory for slipstream caches. Auto-detected if None.
        output_base: Base directory for output. Defaults to cache_base.
        verbose: Print progress info.

    Returns:
        Path to the output cache directory.
    """
    from slipstream.cache import (
        CACHE_VERSION,
        MANIFEST_FILE,
        OptimizedCache,
        _create_field_writer,
        _get_expected_files,
        write_index,
    )

    if subset not in SUBSET_DEFS:
        raise ValueError(f"Unknown subset {subset!r}. Available: {list(SUBSET_DEFS)}")

    sub_def = SUBSET_DEFS[subset]
    subset_name = sub_def["name"]

    # Resolve paths
    source_dir = _resolve_source_cache_dir(split, fmt, cache_base)
    if output_base is None:
        output_base = source_dir.parent
    output_dir = output_base / f"{subset_name}-s256_l512-{fmt}-{split}"

    if verbose:
        print(f"Source cache: {source_dir}")
        print(f"Output cache: {output_dir}")

    # ------------------------------------------------------------------
    # 1. Load source cache
    # ------------------------------------------------------------------
    if not (source_dir / MANIFEST_FILE).exists():
        raise FileNotFoundError(
            f"Source cache not found at {source_dir}. "
            f"Download it first: load('imagenet1k', split='{split}', fmt='{fmt}')"
        )

    source = OptimizedCache.load(source_dir, verbose=verbose)
    if verbose:
        print(f"Source: {source.num_samples} samples, fields: {list(source.field_types)}")

    # ------------------------------------------------------------------
    # 2. Build/load label index and collect subset indices
    # ------------------------------------------------------------------
    try:
        label_index = source.get_index("label")
    except (KeyError, AttributeError):
        if verbose:
            print("Building label index for source cache (one-time)...")
        write_index(source, fields=["label"])
        source = OptimizedCache.load(source_dir, verbose=False)
        label_index = source.get_index("label")

    in1000_classes = sorted(sub_def["in1000_classes"])
    subset_indices = np.sort(np.concatenate([
        label_index[c] for c in in1000_classes
    ]))
    num_subset = len(subset_indices)

    if verbose:
        print(f"Subset {subset}: {num_subset} samples across {len(in1000_classes)} classes")

    # ------------------------------------------------------------------
    # 3. Read source labels for the subset
    # ------------------------------------------------------------------
    source_labels_storage = source.fields["label"]
    all_source_labels = source_labels_storage._data[:]  # mmap read
    subset_in1000_labels = all_source_labels[subset_indices]

    # ------------------------------------------------------------------
    # 4. Write new cache
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine image format from source cache manifest
    source_image_format = source._field_metadata.get("image", {}).get("image_format", fmt)

    # Non-image field types — image is handled separately via raw byte copy
    non_image_fields = {"in1000_label": "int", "path": "str", "index": "int"}
    for label_field_name in sub_def["label_fields"]:
        non_image_fields[label_field_name] = "int"

    writers = {}
    for field_name, field_type in non_image_fields.items():
        writers[field_name] = _create_field_writer(
            field_name, field_type, output_dir, num_subset,
        )

    # --- Raw image copy (no decode/encode) ---
    # Copy image bytes directly from source cache, preserving format as-is.
    # This avoids ImageBytesWriter.add_sample() which tries to detect and
    # transcode image formats (breaks for YUV420 raw bytes).
    from slipstream.cache import VARIABLE_METADATA_DTYPE

    img_storage = source.fields["image"]
    new_img_meta = np.zeros(num_subset, dtype=VARIABLE_METADATA_DTYPE)
    max_img_size = 0
    img_data_file = open(output_dir / "image.bin", "wb")
    current_ptr = 0

    # Access source path storage
    path_storage = source.fields["path"]

    for out_idx, src_idx in enumerate(subset_indices):
        if verbose and out_idx % 5000 == 0 and out_idx > 0:
            print(f"  {out_idx}/{num_subset} samples written...")

        # Image: raw byte copy from source mmap
        src_meta = img_storage._metadata[src_idx]
        ptr = int(src_meta["data_ptr"])
        size = int(src_meta["data_size"])
        img_data_file.write(img_storage._data_mmap[ptr:ptr + size])
        new_img_meta[out_idx]["data_ptr"] = current_ptr
        new_img_meta[out_idx]["data_size"] = size
        new_img_meta[out_idx]["height"] = src_meta["height"]
        new_img_meta[out_idx]["width"] = src_meta["width"]
        current_ptr += size
        max_img_size = max(max_img_size, size)

        # In1000 label (original)
        in1000_label = int(subset_in1000_labels[out_idx])
        writers["in1000_label"].add_sample(out_idx, in1000_label)

        # Subset label fields (remapped)
        for label_field_name, remap in sub_def["label_fields"].items():
            writers[label_field_name].add_sample(out_idx, remap[in1000_label])

        # Path (from source)
        offset, length = path_storage._offsets[src_idx]
        path_str = bytes(path_storage._data_mmap[offset:offset + length]).decode("utf-8")
        writers["path"].add_sample(out_idx, path_str)

        # Index (renumbered)
        writers["index"].add_sample(out_idx, out_idx)

    # Finalize image field
    img_data_file.flush()
    img_data_file.close()
    np.save(output_dir / "image.meta.npy", new_img_meta)
    image_field_meta = {
        "type": "ImageBytes",
        "num_samples": num_subset,
        "max_size": int(max_img_size * 1.2),
        "image_format": source_image_format,
    }

    # ------------------------------------------------------------------
    # 5. Finalize writers and write manifest
    # ------------------------------------------------------------------
    field_metadata = {"image": image_field_meta}
    for field_name, writer in writers.items():
        field_metadata[field_name] = writer.finalize()

    # Compute file sizes for integrity checking
    file_sizes = {}
    for field_name, meta in field_metadata.items():
        for fname in _get_expected_files(field_name, meta["type"]):
            fpath = output_dir / fname
            if fpath.exists():
                file_sizes[fname] = os.path.getsize(fpath)

    manifest = {
        "version": CACHE_VERSION,
        "num_samples": num_subset,
        "fields": field_metadata,
        "file_sizes": file_sizes,
    }
    with open(output_dir / MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"Manifest written: {num_subset} samples, {len(field_metadata)} fields")

    # Free writer memory
    del writers
    import gc
    gc.collect()

    # ------------------------------------------------------------------
    # 6. Build indexes for all label fields
    # ------------------------------------------------------------------
    cache = OptimizedCache.load(output_dir, verbose=verbose)

    index_fields = ["in1000_label"] + list(sub_def["label_fields"].keys())
    # Deduplicate while preserving order
    index_fields = list(dict.fromkeys(index_fields))

    if verbose:
        print(f"Building indexes for: {index_fields}")
    write_index(cache, fields=index_fields)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    if verbose:
        # Reload to pick up indexes
        cache = OptimizedCache.load(output_dir, verbose=False)
        print(f"\nDone! Cache written to: {output_dir}")
        print(f"  Samples: {cache.num_samples}")
        print(f"  Fields:  {list(cache.field_types)}")

        # Per-class distribution using native label
        native_label = "label"
        label_idx = cache.get_index(native_label)
        counts = {k: len(v) for k, v in sorted(label_idx.items())}
        min_c, max_c = min(counts.values()), max(counts.values())
        print(f"  Classes: {len(counts)} (samples per class: min={min_c}, max={max_c})")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build slipstream caches for ImageNet subsets"
    )
    parser.add_argument(
        "--subset", choices=["in100", "in10"],
        help="Subset to build",
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "train"],
        help="Dataset split (default: val)",
    )
    parser.add_argument(
        "--fmt", default="jpeg", choices=["jpeg", "yuv420"],
        help="Image format (default: jpeg)",
    )
    parser.add_argument(
        "--cache-base", type=Path, default=None,
        help="Base directory for slipstream caches (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-base", type=Path, default=None,
        help="Output base directory (defaults to cache-base)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Build all subsets and splits (jpeg only)",
    )
    args = parser.parse_args()

    if args.all:
        for subset in ["in100", "in10"]:
            for split in ["val", "train"]:
                print(f"\n{'='*60}")
                print(f"Building {subset} / {split} / {args.fmt}")
                print(f"{'='*60}")
                build_subset_cache(
                    subset=subset,
                    split=split,
                    fmt=args.fmt,
                    cache_base=args.cache_base,
                    output_base=args.output_base,
                )
    elif args.subset is None:
        parser.error("--subset is required (or use --all)")
    else:
        build_subset_cache(
            subset=args.subset,
            split=args.split,
            fmt=args.fmt,
            cache_base=args.cache_base,
            output_base=args.output_base,
        )


if __name__ == "__main__":
    main()
