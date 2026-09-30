# Proposal: delete the duplicate `slipcache/` copies in the imagenet1k val caches

Status: **proposal, nothing deleted.** Needs George's decision. Written 2026-09-30.

## Finding

Both imagenet1k **val** caches hold a nested `slipcache/` subdir that is a complete second copy of
the cache. Every object in it has the same size and S3 ETag as its top-level twin, and all were
uploaded 2026-04-01.

| cache | top-level cache | `slipcache/` duplicate |
|---|---|---|
| `imagenet1k-s256_l512-jpeg-val` | 3.78 GB | 8 files, 3.78 GB |
| `imagenet1k-s256_l512-yuv420-val` | 6.54 GB | 8 files, 6.54 GB |

The duplicate holds `image.bin`, `image.meta.npy`, `index.npy`, `label.npy`, `label_index.npy`,
`manifest.json`, `path.bin` and `path.offsets.npy`.

No other registered cache has extras. The S3 audit (2026-09-30) covered every `remote_cache` and
video store in the registry: imagenet10/100/100_s292/1k train+val, jpeg+yuv420, and the six
spatialvid-hq stores. The only other unlisted file is a 20-byte `.synced` prep marker in the
spatialvid-hq `h265/456x256/15` store. It's harmless, and sync ignores it.

Nothing reads `slipcache/`. slipstream (0.9.5) loads `<dir>/manifest.json` directly. The
`slipcache/` wording in `download_s3_cache` and `_is_prebuilt_cache` docstrings is stale; neither
function looks inside the subdir. visionlab-datasets >= 0.16.0 doesn't copy it (sync copies only
files the manifest accounts for) and `status` flags it wherever it exists.

## Proposed cleanup (not run)

S3, 10.3 GB:

```bash
s5cmd rm 's3://visionlab-datasets/slipstream-cache/imagenet1k/imagenet1k-s256_l512-jpeg-val/slipcache/*'
s5cmd rm 's3://visionlab-datasets/slipstream-cache/imagenet1k/imagenet1k-s256_l512-yuv420-val/slipcache/*'
```

lab_storage master, reported by model-rearing as the same layout (~10.3 GB). Check it first with
datasets >= 0.16.0, whose `unlisted` field lists exactly what isn't part of each cache:

```bash
SLIPSTREAM_CACHE_DIR=/n/lab_storage/alvarez_lab/Lab/datasets/slipstream \
  visionlab-datasets status --no-remote --json | jq '.datasets[] | select(.unlisted) | {cache_name, unlisted}'
rm -r /n/lab_storage/alvarez_lab/Lab/datasets/slipstream/imagenet1k-s256_l512-jpeg-val/slipcache
rm -r /n/lab_storage/alvarez_lab/Lab/datasets/slipstream/imagenet1k-s256_l512-yuv420-val/slipcache
```

Other copies synced before 0.16.0, e.g. model-rearing's netscratch copy or workstation caches,
carry the same duplicate. `status` flags them as "not in the manifest ... safe to delete".

Before deleting on S3, check that the top-level objects are intact. Their ETags match the
duplicates, and `visionlab-datasets status` passes the size check for a fresh sync of both caches.
