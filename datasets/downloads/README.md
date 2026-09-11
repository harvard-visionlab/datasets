# Raw dataset downloads

Scripts that fetch *raw* releases (archives, metadata) to lab storage. They do not build slipstream caches;
that is `datasets/prep/`. Run them on the machine attached to the target storage (mcp → QNAP).

| Script | Source | Default payload | Target |
| --- | --- | --- | --- |
| `spatialvid_hq.py` | `SpatialVID/SpatialVID-HQ` (HF, gated) | 74 video tars (1.07 TB) + 74 annotation tars (119 GB) + metadata CSV (142 MB); depths (2.33 TB) opt-in | `…/qnap/exactitude/Flash/DataSets/VideoDatasets/SpatialVID-HQ` |

```bash
uv run python -m datasets.downloads.spatialvid_hq --dest <target> --dry-run      # list + sizes
uv run python -m datasets.downloads.spatialvid_hq --dest <target> --types metadata,annotations
uv run python -m datasets.downloads.spatialvid_hq --dest <target>                # everything but depths
uv run python -m datasets.downloads.spatialvid_hq --dest <target> --verify-only  # sizes vs remote; --checksum for sha256
```

Needs a Hugging Face login with the dataset terms accepted (`hf auth login` or `HF_TOKEN`). Resumable.
Archives are kept as `.tar.gz` (no extraction); prep scripts stream them.
