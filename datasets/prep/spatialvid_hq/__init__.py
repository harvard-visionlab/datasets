"""SpatialVID-HQ preparation pipeline (raw HF archives -> slipstream video stores + index + splits).

Stages (each a module with a CLI, all resumable and idempotent):

    build_index   metadata CSV + SpatialVID-RAW source table + per-clip annotations -> index/clips.parquet
    make_splits   source-level stratified train/val assignment            -> splits/<version>.parquet
    encode        per-group re-encode to 640x360 / 456x256 HEVC, GOP 1 s   -> shards/<res>/group_XXXX/
    merge         group shards -> one slipstream cache per resolution       -> stores/spatialvid-hq-h265-<res>/

See README.md in this directory for the data model.
"""
