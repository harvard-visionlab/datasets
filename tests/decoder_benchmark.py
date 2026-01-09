"""
Benchmark different image decoders with StreamingDataset.

Usage:
    uv run python tests/decoder_benchmark.py --decoder turbo --epochs 3
    uv run python tests/decoder_benchmark.py --decoder all --batch-size 128 --num-workers 4

Decoders:
    - turbo: TurboJPEG (fastest CPU decoder)
    - pil: PIL/Pillow
    - cv2: OpenCV
    - torchvision_cpu: torchvision.io.decode_image on CPU
    - torchvision_cuda: torchvision.io.decode_jpeg on CUDA (requires GPU)
    - all: Run all available decoders
"""
import argparse
import io
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image

from visionlab.datasets import StreamingDataset
from litdata.streaming.cache import Dir


# TurboJPEG setup
try:
    from turbojpeg import TurboJPEG, TJPF_RGB
except ImportError:
    try:
        from turbojpeg import TurboJPEG, TJPF
        TJPF_RGB = TJPF.RGB
    except ImportError:
        TurboJPEG = None
        TJPF_RGB = None

# OpenCV
try:
    import cv2
except ImportError:
    cv2 = None

# torchvision decoders
from torchvision.io import decode_image, decode_jpeg, ImageReadMode


@dataclass
class EpochStats:
    epoch: int
    duration_sec: float
    image_count: int
    images_per_sec: float


@dataclass
class BenchmarkResult:
    decoder: str
    batch_size: int
    num_workers: int
    epoch_stats: list[EpochStats]
    avg_images_per_sec: float
    device: str


class DecoderTransform:
    """Base class for decoder transforms that convert bytes to tensor."""

    def __init__(self, crop_size: int = 224):
        self.crop_size = crop_size
        # transforms.v2 works on both PIL images and tensors
        self.to_tensor = transforms.ToImage()
        self.crop = transforms.CenterCrop(crop_size)
        self.to_dtype = transforms.ToDtype(torch.float32, scale=True)

    def __call__(self, image_bytes: bytes) -> torch.Tensor:
        raise NotImplementedError


class TurboDecoder(DecoderTransform):
    """TurboJPEG decoder - fastest CPU option."""

    def __init__(self, crop_size: int = 224):
        super().__init__(crop_size)
        if TurboJPEG is None:
            raise ImportError("TurboJPEG not available")
        self.turbo = TurboJPEG()

    def __call__(self, image_bytes: bytes) -> torch.Tensor:
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.tobytes()
        rgb_array = self.turbo.decode(image_bytes, pixel_format=TJPF_RGB)
        tensor = self.to_tensor(rgb_array)
        tensor = self.crop(tensor)
        return self.to_dtype(tensor)


class PILDecoder(DecoderTransform):
    """PIL/Pillow decoder."""

    def __init__(self, crop_size: int = 224):
        super().__init__(crop_size)

    def __call__(self, image_bytes: bytes) -> torch.Tensor:
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.tobytes()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = self.to_tensor(pil_image)
        tensor = self.crop(tensor)
        return self.to_dtype(tensor)


class CV2Decoder(DecoderTransform):
    """OpenCV decoder."""

    def __init__(self, crop_size: int = 224):
        super().__init__(crop_size)
        if cv2 is None:
            raise ImportError("OpenCV (cv2) not available")

    def __call__(self, image_bytes: bytes) -> torch.Tensor:
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.tobytes()
        bgr_image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = self.to_tensor(rgb_image)
        tensor = self.crop(tensor)
        return self.to_dtype(tensor)


class TorchvisionCPUDecoder(DecoderTransform):
    """torchvision.io.decode_image on CPU."""

    def __init__(self, crop_size: int = 224):
        super().__init__(crop_size)

    def __call__(self, image_bytes: bytes) -> torch.Tensor:
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.tobytes()
        img_buffer = torch.frombuffer(image_bytes, dtype=torch.uint8)
        # decode_image returns CHW tensor with uint8
        tensor = decode_image(img_buffer, mode=ImageReadMode.RGB)
        tensor = self.crop(tensor)
        return self.to_dtype(tensor)


class TorchvisionCUDADecoder(DecoderTransform):
    """torchvision.io.decode_jpeg on CUDA - requires GPU."""

    def __init__(self, crop_size: int = 224):
        super().__init__(crop_size)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for torchvision_cuda decoder")

    def __call__(self, image_bytes: bytes) -> torch.Tensor:
        if isinstance(image_bytes, np.ndarray):
            image_bytes = image_bytes.tobytes()
        img_buffer = torch.frombuffer(image_bytes, dtype=torch.uint8)
        # decode_jpeg with device='cuda' returns tensor on GPU
        tensor = decode_jpeg(img_buffer, device='cuda')
        tensor = self.crop(tensor)
        return self.to_dtype(tensor)


DECODERS = {
    'turbo': TurboDecoder,
    'pil': PILDecoder,
    'cv2': CV2Decoder,
    'torchvision_cpu': TorchvisionCPUDecoder,
    'torchvision_cuda': TorchvisionCUDADecoder,
}


def get_available_decoders() -> list[str]:
    """Return list of decoders available on this system."""
    available = []

    for name, decoder_cls in DECODERS.items():
        try:
            decoder_cls()
            available.append(name)
        except (ImportError, RuntimeError) as e:
            print(f"  Decoder '{name}' not available: {e}")

    return available


def create_decoder(name: str, crop_size: int = 224) -> DecoderTransform:
    """Create a decoder instance by name."""
    if name not in DECODERS:
        raise ValueError(f"Unknown decoder: {name}. Available: {list(DECODERS.keys())}")
    return DECODERS[name](crop_size=crop_size)


def run_benchmark(
    remote_dir: str,
    cache_dir: str | None,
    decoder_name: str,
    batch_size: int,
    num_workers: int,
    epochs: int,
) -> BenchmarkResult:
    """Run benchmark for a single decoder."""

    print(f"\n{'='*60}")
    print(f"Benchmarking decoder: {decoder_name}")
    print(f"{'='*60}")

    decoder = create_decoder(decoder_name)
    device = 'cuda' if decoder_name == 'torchvision_cuda' else 'cpu'

    # Setup dataset with decoder transform
    if cache_dir:
        input_dir = Dir(path=cache_dir, url=remote_dir)
    else:
        input_dir = remote_dir

    ds = StreamingDataset(
        input_dir=input_dir,
        decode_images=False,  # We handle decoding ourselves
        pipelines=dict(image=decoder),
    )

    print(f"Dataset: {len(ds)} images")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )

    epoch_stats = []

    for epoch in range(epochs):
        start_time = time.perf_counter()
        image_count = 0

        for batch in dl:
            if isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch[0]

            # Move to device if using CUDA decoder (images already on GPU)
            if device == 'cuda' and images.device.type != 'cuda':
                images = images.to('cuda', non_blocking=True)

            image_count += images.shape[0]

        duration = time.perf_counter() - start_time
        images_per_sec = image_count / duration

        stats = EpochStats(
            epoch=epoch + 1,
            duration_sec=round(duration, 2),
            image_count=image_count,
            images_per_sec=round(images_per_sec, 1),
        )
        epoch_stats.append(stats)

        print(f"  Epoch {epoch + 1}: {duration:.2f}s, {image_count} images, {images_per_sec:.1f} img/s")

    avg_images_per_sec = sum(s.images_per_sec for s in epoch_stats) / len(epoch_stats)

    return BenchmarkResult(
        decoder=decoder_name,
        batch_size=batch_size,
        num_workers=num_workers,
        epoch_stats=epoch_stats,
        avg_images_per_sec=round(avg_images_per_sec, 1),
        device=device,
    )


def print_summary(results: list[BenchmarkResult]):
    """Print summary table of all benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Decoder':<20} {'Device':<8} {'Avg img/s':>12} {'Epochs':>8}")
    print("-"*70)

    # Sort by average images per second (descending)
    sorted_results = sorted(results, key=lambda r: r.avg_images_per_sec, reverse=True)

    for r in sorted_results:
        print(f"{r.decoder:<20} {r.device:<8} {r.avg_images_per_sec:>12.1f} {len(r.epoch_stats):>8}")

    print("-"*70)

    if len(sorted_results) > 1:
        fastest = sorted_results[0]
        print(f"\nFastest decoder: {fastest.decoder} ({fastest.avg_images_per_sec:.1f} img/s)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark image decoders with StreamingDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--remote-dir",
        default="s3://visionlab-datasets/imagenet100/pre-processed/s256-l512-jpgbytes-q100-streaming/val",
        help="Remote S3 path to streaming dataset",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local cache directory (optional)",
    )
    parser.add_argument(
        "--decoder",
        default="all",
        choices=list(DECODERS.keys()) + ["all"],
        help="Decoder to benchmark (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to run (default: 3)",
    )

    args = parser.parse_args()

    print("Decoder Benchmark")
    print("="*60)
    print(f"Remote dir: {args.remote_dir}")
    print(f"Cache dir: {args.cache_dir or '(none)'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Epochs: {args.epochs}")

    # Determine which decoders to run
    if args.decoder == "all":
        print("\nChecking available decoders...")
        decoders_to_run = get_available_decoders()
        print(f"Will benchmark: {decoders_to_run}")
    else:
        decoders_to_run = [args.decoder]

    if not decoders_to_run:
        print("No decoders available!")
        return

    results = []
    for decoder_name in decoders_to_run:
        try:
            result = run_benchmark(
                remote_dir=args.remote_dir,
                cache_dir=args.cache_dir,
                decoder_name=decoder_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epochs=args.epochs,
            )
            results.append(result)
        except Exception as e:
            print(f"\nError benchmarking {decoder_name}: {e}")

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
