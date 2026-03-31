#!/usr/bin/env python3
"""Benchmark visionlab.datasets loader performance.

Loads a registered dataset and runs full forward passes through
SlipstreamLoader, reporting images/sec throughput.

Usage:
    uv run python benchmarks/benchmark_loader.py
    uv run python benchmarks/benchmark_loader.py --dataset imagenet1k --split val --fmt jpeg
    uv run python benchmarks/benchmark_loader.py --dataset imagenet1k --split val --fmt yuv420
    uv run python benchmarks/benchmark_loader.py --epochs 3 --batch-size 512
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from tqdm import tqdm


def benchmark(
    dataset_name: str,
    split: str,
    fmt: str,
    batch_size: int,
    num_epochs: int,
    num_warmup: int,
    pipeline: str,
    target_size: int,
    num_threads: int,
):
    from visionlab.datasets import load
    from slipstream import SlipstreamLoader
    from slipstream.pipelines import supervised_train, supervised_val

    print(f"Dataset: {dataset_name} (split={split}, fmt={fmt})")
    print(f"Pipeline: {pipeline} (size={target_size})")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_threads or 'auto'}")
    print()

    # Load dataset (downloads from S3 if needed)
    dataset = load(dataset_name, split=split, fmt=fmt)
    print(f"Samples: {len(dataset):,}")

    # Select pipeline
    if pipeline == "val":
        pipelines = supervised_val(target_size, num_threads=num_threads)
    elif pipeline == "train":
        pipelines = supervised_train(target_size, num_threads=num_threads)
    else:
        pipelines = None

    loader = SlipstreamLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pipelines=pipelines,
        exclude_fields=["path"],
    )

    def run_epoch():
        total = 0
        for batch in tqdm(loader, leave=False):
            if pipelines is None:
                img = batch.get("image")
                if isinstance(img, dict):
                    total += len(img["data"])
                else:
                    total += batch_size
            else:
                total += batch["image"].shape[0]
        return total

    # Warmup
    print(f"\nWarmup ({num_warmup} epoch(s)):")
    for i in range(num_warmup):
        t0 = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - t0
        rate = total / elapsed
        print(f"  Warmup {i + 1}: {rate:,.0f} img/s ({elapsed:.2f}s)")

    # Timed epochs
    rates = []
    print(f"\nBenchmark ({num_epochs} epoch(s)):")
    for i in range(num_epochs):
        t0 = time.perf_counter()
        total = run_epoch()
        elapsed = time.perf_counter() - t0
        rate = total / elapsed
        rates.append(rate)
        print(f"  Epoch {i + 1}: {rate:,.0f} img/s ({elapsed:.2f}s)")

    avg = np.mean(rates)
    std = np.std(rates) if len(rates) > 1 else 0
    print(f"\nResult: {avg:,.0f} ± {std:,.0f} img/s")

    loader.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark visionlab.datasets loader")
    parser.add_argument("--dataset", type=str, default="imagenet1k",
                        help="Dataset name (default: imagenet1k)")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split (default: val)")
    parser.add_argument("--fmt", type=str, default="jpeg",
                        choices=["jpeg", "yuv420"],
                        help="Image format (default: jpeg)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of timed epochs (default: 3)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup epochs (default: 1)")
    parser.add_argument("--pipeline", type=str, default="val",
                        choices=["val", "train", "raw"],
                        help="Pipeline type (default: val)")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Crop size (default: 224)")
    parser.add_argument("--num-threads", type=int, default=0,
                        help="Decoder threads, 0=auto (default: 0)")
    args = parser.parse_args()

    benchmark(
        dataset_name=args.dataset,
        split=args.split,
        fmt=args.fmt,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_warmup=args.warmup,
        pipeline=args.pipeline,
        target_size=args.target_size,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
