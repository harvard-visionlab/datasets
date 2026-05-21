"""Tests for normalization-stats attachment in ``load()``.

Normalization stats depend on the DECODED colorspace, not the storage ``fmt``.
Every ``Decode*`` path emits RGB pixels, so ``.stats`` defaults to the rgb
stats regardless of ``fmt``. The raw-YUV stats remain available (only correct
for the niche ``DecodeYUV*`` consumers) via ``.stats_by_colorspace``.
"""
import pytest

from visionlab.datasets.registry import get_config, load

# Stats of the decoded RGB output (the common decode path).
RGB_STATS = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
# Stats of the raw YUV planes (only correct for DecodeYUV* consumers).
YUV_STATS = {
    "mean": (0.455585, 0.470487, 0.515044),
    "std": (0.264127, 0.068042, 0.064571),
}

# Datasets that define a metadata["stats"] dict.
STATS_DATASETS = ["imagenet10", "imagenet100", "imagenet100_s292", "imagenet1k"]


@pytest.fixture
def fake_load(monkeypatch, tmp_path):
    """Patch out the S3 download + SlipstreamDataset so ``load()`` runs offline.

    Returns a helper that pre-creates the local cache manifest for a given
    (name, split, fmt) so ``load()`` skips the download path entirely.
    """
    import slipstream
    from slipstream.cache import MANIFEST_FILE
    import visionlab.datasets.registry as registry

    class FakeDataset:
        def __init__(self, local_dir, **kwargs):
            self.local_dir = local_dir
            self.kwargs = kwargs

    monkeypatch.setattr(slipstream, "SlipstreamDataset", FakeDataset)
    monkeypatch.setattr(
        registry, "configure_slipstream_cache", lambda: str(tmp_path)
    )

    def _prepare(name, split, fmt):
        config = get_config(name)
        remote = config.remote_cache[(split, fmt)]
        cache_name = remote.rstrip("/").rsplit("/", 1)[-1]
        cache_dir = tmp_path / cache_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = cache_dir / MANIFEST_FILE
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")

    return _prepare


@pytest.mark.parametrize("name", STATS_DATASETS)
def test_yuv420_load_returns_rgb_stats(name, fake_load):
    """load(..., fmt='yuv420').stats returns RGB stats (the decoded output)."""
    fake_load(name, "val", "yuv420")
    ds = load(name, split="val", fmt="yuv420")
    assert ds.stats == RGB_STATS


@pytest.mark.parametrize("name", STATS_DATASETS)
def test_jpeg_load_returns_rgb_stats(name, fake_load):
    """load(..., fmt='jpeg').stats returns RGB stats (the decoded output)."""
    fake_load(name, "val", "jpeg")
    ds = load(name, split="val", fmt="jpeg")
    assert ds.stats == RGB_STATS


@pytest.mark.parametrize("name", STATS_DATASETS)
@pytest.mark.parametrize("fmt", ["jpeg", "yuv420"])
def test_stats_by_colorspace_exposes_all(name, fmt, fake_load):
    """stats_by_colorspace keeps rgb + yuv420 regardless of storage fmt."""
    fake_load(name, "val", fmt)
    ds = load(name, split="val", fmt=fmt)
    assert ds.stats_by_colorspace["rgb"] == RGB_STATS
    assert ds.stats_by_colorspace["yuv420"] == YUV_STATS
