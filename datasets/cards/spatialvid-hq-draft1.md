# SpatialVID-HQ (lab build) — dataset card

Registry name `spatialvid-hq` · `load("spatialvid-hq", split=, subset=, rate_hz=|fps=, res=, where=, channel_cap=)` → `VideoDataset`
· lab tree `DataSets/VideoDatasets/SpatialVID-HQ-slipstream/` (QNAP exactitude Flash) · S3 `s3://visionlab-datasets/slipstream-cache/spatialvid-hq/`
· card written 2026-09-18 from `index/summary.json`, `stores/*/store_manifest.json`, `subsets/person_carried_v0.report.md`, `splits/v3.report.md`.

## 1. Source, license, citation

- **Upstream**: SpatialVID-HQ, the curated high-quality subset of SpatialVID (Nanjing University, 2025): YouTube clips with
  per-frame camera poses, depth-derived annotations, captions and motion/scene tags. HF: `SpatialVID/SpatialVID-HQ`.
- **License**: the upstream dataset card lists CC BY-NC-SA 4.0 for the annotations; the videos remain the YouTube uploaders'
  (research use only, no redistribution of frames). Verify against the HF card before any external release.
- **Cite**: Wang et al., *SpatialVID: A Large-Scale Video Dataset with Spatial Annotations*, 2025 (arXiv:2509.09676).
- **Lab build**: `visionlab-datasets` (`datasets/prep/spatialvid_hq/`), slipstream ≥ 0.7.0. Store build dates: native stores
  2026-09-13, fps stores 2026-09-17/18. Split v3 2026-09-17.

## 2. What a record is

One upstream clip (a shot of a YouTube video, 2–~30 s), stored as an h265 MP4 byte string plus its annotations:

| field | meaning |
| --- | --- |
| `video` | h265 MP4 (libx265, crf 29, preset medium, 1 s GOP), one of the resolutions below |
| `clip_id`, `source_id`, `group_id` | upstream ids: clip, YouTube video (source), HQ shard |
| `fps`, `num_frames`, `duration_s`, `width`, `height` | of the stored stream (fps stores: decimated); `src_fps`, `src_num_frames` = source stream |
| `poses` | (n, 7) world→camera `[tx ty tz qx qy qz qw]`, OpenCV axes (x right, y down, z forward), **non-metric scale**, ≈ 5 Hz |
| `annot_frame_idx` | (n,) *source* frame index of each pose row (time = idx / `src_fps`) |
| `intrinsics` | (n, 4) normalised `[fx fy cx cy]` per annotated frame |
| `caption`, `scene_type`, `time_of_day`, `weather`, `crowd_density`, `motion_tags`, `instructions`, scores | upstream text / tags / quality scores (captions are noisy) |

Poses are interpolated to arbitrary frame times by `VideoDataset.poses_at` (linear position, slerp rotation); frame-to-frame
ego-motion `[dx dy dz rx ry rz]` in the previous frame's camera axes by `VideoDataset.ego_motion_at` / `video.ego_motion`.

## 3. Counts

| | clips | hours | frames | sources (videos) | channels |
| --- | ---: | ---: | ---: | ---: | ---: |
| upstream HQ index (`index/clips.parquet`) | 365,362 | 1,112 | — | 22,543 | — |
| **stores** (every clip with consistent annotations) | **365,296** | | | | |
| population `person_carried_v0` | 213,103 (58.3 %) | 644 | 106.7 M | 13,222 | 82 |

Source frame rates in the population: 60 fps 51 %, 30 fps 38 %, 24 fps 6 %, 50 fps 3 %, 25 fps 2 %. Duration: 68 % of the
population clips (87 % of frames) are ≥ 8 s; 88 % ≥ 4 s.

### Counts per subset × split (split version v3; hours/frames from `index/clips.parquet`)

`load(split=...)` returns the **population** rows by default; `subset="all"` the whole-store rows.

**Population `person_carried_v0`** (`load(..., split=)`)

| split | clips | videos | channels | hours | frames (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 186,710 | 11,745 | 75 | 563 | 93.0 |
| val | 15,029 | 729 | 63 | 46 | 7.5 |
| test | 11,364 | 748 | 7 | 35 | 6.2 |
| all | 213,103 | 13,222 | 82 | 644 | 106.7 |

| carrier | train | val | test | total |
| --- | ---: | ---: | ---: | ---: |
| walk | 163,164 | 13,185 | 11,364 | 187,713 |
| rig | 23,546 | 1,844 | 0 | 25,390 |

**Whole store** (`load(..., split=, subset="all")`)

| split | clips | videos | channels | hours | frames (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 334,494 | 20,939 | 129 | 1,020 | 169.1 |
| val | 16,932 | 729 | 63 | 50 | 8.2 |
| test | 13,870 | 872 | 7 | 42 | 7.2 |
| excluded (not in any store) | 66 | 48 | 35 | 0 | 0.0 |
| all | 365,362 | 22,543 | 136 | 1,112 | 184.5 |

Clips outside the population, by carrier (what `subset="all"` adds; `walk`/`rig` rows here failed the speed or
stationary-tag criteria):

| carrier | train | val | test | excluded | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| drive | 50,734 | 0 | 489 | 2 | 51,225 |
| mixed | 22,539 | 0 | 242 | 8 | 22,789 |
| walk | 18,840 | 1,507 | 1,499 | 41 | 21,887 |
| no_channel | 11,025 | 0 | 0 | 1 | 11,026 |
| bike | 9,939 | 0 | 46 | 1 | 9,986 |
| drone | 7,983 | 0 | 0 | 4 | 7,987 |
| rig | 7,290 | 396 | 0 | 7 | 7,693 |
| train | 7,457 | 0 | 44 | 0 | 7,501 |
| unlabelled | 5,686 | 0 | 0 | 1 | 5,687 |
| other | 5,519 | 0 | 0 | 1 | 5,520 |
| boat | 772 | 0 | 186 | 0 | 958 |

Val videos are chosen from channels that are in train and stratified to the population, so non-population val rows are
only the walk/rig clips of those videos that failed the speed/stationary criteria. The 7 test channels are walking
channels, so their non-population rows are mostly the same plus a few drive/boat/bike clips those creators uploaded.
Regenerate these tables with `python -m datasets.prep.spatialvid_hq.card_counts --tree <tree>`.

## 4. Stores (fmt × resolution × fps)

| store | frames | bytes | notes |
| --- | --- | ---: | --- |
| `spatialvid-hq-h265-640x360` | source rate | 296 GB | native |
| `spatialvid-hq-h265-456x256` | source rate | 169 GB (178,454,929,460 B video) | native, **default resolution** |
| `…-640x360-30fps`, `…-456x256-30fps` | 30 fps | 311 / 176 GB | integer decimation of 60 fps sources, 30/24/25 fps kept as is |
| `…-640x360-15fps`, `…-456x256-15fps` | 15 fps | 295 / 167 GB | 60 → 15, 30 → 15, 50 → 16.7, 24/25 kept |

fps stores: decimation `select=not(mod(n,k))` with exact timestamps (`setpts=N*k/FRAME_RATE/TB`), so frame times are exact
and `annot_frame_idx` still indexes *source* frames; ~0.03 % VFR sources use a select-only fallback. `rate_hz` in `load()` picks
the sparsest store whose fps is an integer multiple of the rate (15 Hz → 15 fps, 10/30 Hz → 30 fps); `fps=` picks exactly;
`fps="native"` the un-decimated store. Default rate 15 Hz. All six stores contain the full 365,296-clip set (a patch pass re-encodes any failed clip).

## 5. Exclusions

66 upstream clips (`index/excluded_clips.csv`) are in no store: their annotation files are internally inconsistent (pose /
intrinsics row count ≠ frame-index row count), so no frame can be assigned a pose. They carry `split = excluded` in the split table.

## 6. Population (subset) `person_carried_v0`

Definition (`subsets/person_carried_v0.report.md`): carrier ∈ {walk, rig} — a **person-borne camera** (walking / hand-held /
head-mounted; `rig` = stabiliser or chest rig) — no `stationary` motion tag, speed = `move_dist / duration_s` ≤ 0.5 (pose units/s),
clip present in the stores. Carrier comes from a channel-level review of the 82 channels plus per-keyword video groups
(`index/carrier_v1.parquet`; 86 % of clips inherit the channel label). Excluded carriers: drone, drive, bike, boat, train, tripod/interior tours.
walk 187,713 · rig 25,390.

**Channel concentration** (a known bias): top channel 14.8 % of clips (Rain Everyday), top 5 = 36.7 %, top 10 ≈ 54 %. Use
`channel_cap=` in `load()` to thin dominant channels, and never evaluate on train channels (see the split).

## 7. Splits (version v3, `splits/v3.parquet` + `v3.report.md`)

The split table labels **every** index clip (365,362 rows; columns `split`, `val_kind`, `channel_id`, `carrier`, `in_subset`,
strata) so any subset can reuse it; the numbers below are the population (`in_subset`), which `load()` applies by default
(`subset="all"` for the whole store).

| split | unit | clips | sources | purpose |
| --- | --- | ---: | ---: | --- |
| train | — | 186,710 | 11,745 | |
| val | whole YouTube **videos** of channels that are in train, stratified to train (≤ 1.2 pp off per stratum) | 15,029 | 729 | in-distribution model selection |
| test | 7 whole **channels** never seen in train, typical strata mix, walk only | 11,364 | 748 | transfer to unseen creators |

Test channels: 4K Nature and City Walks, The Flying Dutchman, Justwalk, TokyoNinjaWalk, Drifted Films, Hui Chen, Trillionex Travel.
Whole-store counts of the same labels: train 334,494 · val 16,932 · test 13,870 · excluded 66. Built with
`make_splits --val-unit three --test-clips 11000 --val-clips 15000 --max-source-clips 200 --max-unit-frac 0.015`. v1 (source-level, no
channel holdout) is kept for comparison only.

## 8. Conventions and caveats

- Poses: world→camera, OpenCV axes, scale is per-clip and non-metric (SfM); compare motion *within* a clip only, or normalise.
- Pose times are `annot_frame_idx / src_fps`; `poses_at` clamps outside the annotated range (first/last ~0.1 s of a clip).
- `DecodeVideoWindow` never requests the last two frames of a clip (NVDEC last-frame bug); `end_margin_frames` in the sampler.
- torchcodec `get_frames_played_at(t)` returns the frame *playing at* t (floor); the loader returns the true pts, use those.
- Decode is CPU-bound: 64 decoders on machina give ~163 windows/s (T = 120, 15 Hz, resize 224) from the 15 fps store, 139 from
  30 fps, 121 from native; off a CIFS mount the first epoch is disk-bound (84) and later epochs match (page cache). Stage the store
  in `$SLIPSTREAM_CACHE_DIR`, set `OMP_NUM_THREADS=1` in every process hosting the stage.
- Normalisation stats: `metadata["stats"][<store>]` in `_configs/spatialvid_hq.py` (RGB, frame-level, computed by `prep/spatialvid_hq/stats.py`).
- Captions and scene/weather tags are model-generated upstream and noisy; the carrier labels are lab-reviewed at channel level, not per clip.

## 9. Versions

| item | value |
| --- | --- |
| index build | 2026-09-12 (`index/summary.json`) |
| stores | native 2026-09-13; 30/15 fps 2026-09-17/18 (`store_manifest.json`, `errors` = clips that needed the patch pass) |
| subset | `person_carried_v0` (2026-09-16 definition) |
| split | v3 (2026-09-17) |
| software | visionlab-datasets 0.9.0, slipstream 0.7.0, ffmpeg 4.4.2 (machina) / 6.1.1 (fleet), libx265 |
