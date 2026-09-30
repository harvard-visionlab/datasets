"""Tests for the ``visionlab-datasets`` CLI.

Offline: S3 and download calls are patched; the cache dir is a tmp_path with
fake caches (manifest + integrity patched) so local checks run for real.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import visionlab.datasets.cli as cli

MANIFEST = "manifest.json"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "given, expected",
    [
        ("in10", "imagenet10"),
        ("IN100", "imagenet100"),
        ("in1k", "imagenet1k"),
        ("in1000", "imagenet1k"),
        ("in100_s292", "imagenet100_s292"),
        ("in100-s292", "imagenet100_s292"),
        ("imagenet100_s292", "imagenet100_s292"),
        ("imagenet10", "imagenet10"),
    ],
)
def test_resolve_name(given, expected):
    assert cli.resolve_name(given) == expected


def test_resolve_name_unknown_lists_options():
    with pytest.raises(SystemExit) as ei:
        cli.resolve_name("nope")
    msg = str(ei.value)
    assert "imagenet100" in msg and "in100=imagenet100" in msg


def test_all_aliases_point_at_registered_datasets():
    from visionlab.datasets import list_datasets

    assert set(cli.ALIASES.values()) <= set(list_datasets())
    assert set(cli.PRIMARY_ALIASES) <= set(cli.ALIASES)


@pytest.mark.parametrize(
    "value, default, expected",
    [
        (None, ["val"], ["val"]),
        ("all", ["val"], ["train", "val"]),
        ("train,val", ["val"], ["train", "val"]),
        ("val,train", ["val"], ["val", "train"]),
        (" val , val ", ["val"], ["val"]),
        ("TRAIN", ["val"], ["train"]),
    ],
)
def test_parse_list_splits(value, default, expected):
    assert cli.parse_list(value, cli.SPLITS, default) == expected


def test_parse_list_rejects_unknown():
    with pytest.raises(SystemExit):
        cli.parse_list("test", cli.SPLITS, ["val"])


def test_cache_name_matches_registry_load_rule():
    from visionlab.datasets import get_config

    remote = get_config("imagenet100_s292").remote_cache[("val", "jpeg")]
    assert cli.cache_name_for(remote) == "imagenet100-s292_l584-jpeg-val"
    assert cli.cache_name_for(remote + "/") == "imagenet100-s292_l584-jpeg-val"


def test_fmt_bytes():
    assert cli.fmt_bytes(None) == "?"
    assert cli.fmt_bytes(512) == "512 B"
    assert cli.fmt_bytes(34_900_000) == "33.3 MB"


# --------------------------------------------------------------------------- #
# fixtures: fake cache dir + patched slipstream plumbing
# --------------------------------------------------------------------------- #


@dataclass
class FakeS3Info:
    s5cmd_path: str | None = "/usr/bin/s5cmd"
    s5cmd_version: str | None = "v2"
    s5cmd_error: str | None = None
    credentials_found: bool = True
    credentials_method: str | None = "env"
    profile: str | None = None
    region: str | None = "us-east-1"
    identity_arn: str | None = "arn:aws:iam::****:user/test"
    identity_error: str | None = None
    bucket_url: str | None = None
    bucket_readable: bool | None = True
    bucket_error: str | None = None
    checked: bool = True


class FakePlumbing:
    """Stands in for slipstream.cli; inspect_dir/find_other_caches are the real ones."""

    def __init__(self):
        import slipstream.cli as real

        self.inspect_dir = real.inspect_dir
        self.find_other_caches = real.find_other_caches
        self.remote_sizes: dict[str, tuple[int, int]] = {}
        self.remote_error: Exception | None = None
        self.s3 = FakeS3Info()
        self.listed: list[str] = []

    def check_s3(self, remote_base, *, check_remote=True, endpoint_url=None):
        self.s3.bucket_url = remote_base
        self.s3.checked = check_remote
        return self.s3

    def remote_listing(self, remote, *, endpoint_url=None, profile=None):
        self.listed.append(remote)
        if self.remote_error is not None:
            raise self.remote_error
        return self.remote_sizes.get(remote, (14, 889_000_000))


def make_cache(root: Path, cache_name: str, num_samples: int = 5000, nbytes: int = 1000) -> Path:
    d = root / cache_name
    d.mkdir(parents=True)
    (d / MANIFEST).write_text(json.dumps({"num_samples": num_samples}))
    (d / "data.bin").write_bytes(b"x" * nbytes)
    return d


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Cache dir -> tmp_path; slipstream plumbing faked; integrity always ok."""
    from slipstream.cache import OptimizedCache

    monkeypatch.setenv("SLIPSTREAM_CACHE_DIR", str(tmp_path))
    plumbing = FakePlumbing()
    monkeypatch.setattr(cli, "_slipstream_cli", lambda: plumbing)
    monkeypatch.setattr(OptimizedCache, "check_integrity", staticmethod(fake_integrity))
    fake = FakeS3()
    monkeypatch.setattr(cli.cache_sync, "list_remote_files", fake.list)
    monkeypatch.setattr(cli.cache_sync, "fetch_files", fake.fetch)
    plumbing.s3fake = fake
    plumbing.downloads = fake.fetched
    plumbing.root = tmp_path
    return plumbing


def fake_integrity(path):
    """Like slipstream's size check, for fake manifests that list ``_files`` {name: size}."""
    try:
        files = json.loads((Path(path) / MANIFEST).read_text()).get("_files", {})
    except (OSError, ValueError):
        return False, ["manifest.json missing"]
    probs = [f"missing: {n}" for n, sz in files.items()
             if not (Path(path) / n).exists() or (Path(path) / n).stat().st_size != sz]
    return not probs, probs


def fake_manifest(files: dict[str, bytes], **extra) -> bytes:
    return json.dumps({"num_samples": 99, "_files": {n: len(b) for n, b in files.items()}, **extra}).encode()


class FakeS3:
    """Remote caches as {remote: {name: bytes}}; unknown remotes get a small default cache."""

    def __init__(self):
        self.objects: dict[str, dict[str, bytes]] = {}
        self.error: Exception | None = None
        self.corrupt: set[str] = set()  # names fetched with right size, wrong content
        self.fail_after: int | None = None  # fetch this many files, then report failure
        self.fetched: list[tuple[str, Path, list[str]]] = []  # (remote, dest, names)
        self.listed: list[str] = []

    def remote(self, remote: str) -> dict[str, bytes]:
        if remote not in self.objects:
            data = {"image.bin": b"i" * 3000, "image.meta.npy": b"m" * 100, "label.npy": b"l" * 50}
            self.objects[remote] = {MANIFEST: fake_manifest(data), **data}
        return self.objects[remote]

    def list(self, remote, *, endpoint_url=None, profile=None):
        self.listed.append(remote)
        if self.error is not None:
            raise self.error
        return {n: len(b) for n, b in self.remote(remote).items()}

    def fetch(self, remote, names, dest, *, total_bytes, endpoint_url=None, numworkers=32,
              concurrency=None, part_size_mb=None, log=print):
        self.fetched.append((remote, Path(dest), list(names)))
        objs = self.remote(remote)
        for i, n in enumerate(names):
            if self.fail_after is not None and n != MANIFEST and i >= self.fail_after:
                return False
            data = objs[n]
            if n in self.corrupt:
                data = b"X" * len(data)
            (Path(dest) / n).write_bytes(data)
        return True


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def test_status_table_and_paths(env, capsys):
    make_cache(env.root, "imagenet10-s256_l512-jpeg-val", num_samples=5000)
    make_cache(env.root, "slipcache-deadbeef")  # not in registry
    rc = cli.main(["status", "--paths", "--no-remote"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"path        {env.root}" in out
    assert "source      SLIPSTREAM_CACHE_DIR environment variable" in out
    assert "identity    -  (remote checks skipped)" in out
    # present entry, missing entry, remote unchecked
    assert "imagenet10        val    jpeg    ✓" in out
    assert "imagenet10        train  jpeg    ✗ missing" in out
    assert "to fetch:   visionlab-datasets sync imagenet10 val yuv420" in out  # first missing entry
    assert "registered but no remote caches yet: imagenette" in out
    assert "Local cache paths" in out
    assert str(env.root / "imagenet10-s256_l512-jpeg-val") in out
    assert str(env.root / "imagenet10-s256_l512-jpeg-train") not in out
    assert "Other slipstream caches in cache dir" in out and "slipcache-deadbeef" in out
    assert "aliases: in10=imagenet10, in100=imagenet100" in out
    assert "✓ Everything looks good." in out


def test_status_sample_count_mismatch_warns(env, capsys):
    make_cache(env.root, "imagenet100-s292_l584-jpeg-val", num_samples=4999)
    cli.main(["status", "--no-remote"])
    out = capsys.readouterr().out
    assert "⚠ imagenet100-s292_l584-jpeg-val: sample count 4,999 != registry num_val 5,000" in out


def test_status_remote_checks(env, capsys):
    env.remote_sizes = {}
    rc = cli.main(["status"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "identity    ✓  arn:aws:iam::****:user/test" in out
    assert "read        ✓  s3://visionlab-datasets/slipstream-cache/" in out
    assert "✓  847.8 MB" in out  # 889_000_000 bytes
    assert len(env.listed) == 16  # every registered (split, fmt)


def test_status_no_credentials_is_a_problem(env, capsys):
    env.s3 = FakeS3Info(credentials_found=False, identity_arn=None, bucket_readable=False, bucket_error="no credentials")
    rc = cli.main(["status"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Problems" in out and "No AWS credentials found" in out
    assert env.listed == []  # no remote listing without creds


def test_status_json(env, capsys):
    rc = cli.main(["status", "--json", "--no-remote"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data["visionlab_datasets_version"] == cli.__version__
    assert data["cache"]["path"] == str(env.root)
    assert data["aliases"] == {"in10": "imagenet10", "in100": "imagenet100", "in1k": "imagenet1k", "in100_s292": "imagenet100_s292"}
    assert {d["dataset"] for d in data["datasets"]} == {"imagenet10", "imagenet100", "imagenet100_s292", "imagenet1k"}
    assert data["datasets_without_caches"] == ["imagenette"]


def test_status_missing_cache_dir_is_soft(env, capsys, monkeypatch):
    monkeypatch.setenv("SLIPSTREAM_CACHE_DIR", str(env.root / "not-yet"))
    rc = cli.main(["status", "--no-remote"])
    out = capsys.readouterr().out
    assert rc == 0  # creatable -> not a hard failure
    assert "does not exist yet (will be created on first sync)" in out


# --------------------------------------------------------------------------- #
# list / path
# --------------------------------------------------------------------------- #


def test_list_text(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "imagenet100  (100 classes)  [in100]" in out
    assert "imagenette  (10 classes)" in out
    assert "(no remote caches registered)" in out
    assert "s3://visionlab-datasets/slipstream-cache/imagenet100/imagenet100-s292_l584-jpeg-val" in out


def test_path_prints_local_paths(env, capsys):
    make_cache(env.root, "imagenet10-s256_l512-jpeg-val")
    assert cli.main(["path", "in10", "all"]) == 0  # default: all formats
    out = capsys.readouterr().out.splitlines()
    assert out == [  # SPLITS x FMTS order
        f"✗ imagenet10 train jpeg  {env.root / 'imagenet10-s256_l512-jpeg-train'}",
        f"✗ imagenet10 train yuv420  {env.root / 'imagenet10-s256_l512-yuv420-train'}",
        f"✓ imagenet10 val jpeg  {env.root / 'imagenet10-s256_l512-jpeg-val'}",
        f"✗ imagenet10 val yuv420  {env.root / 'imagenet10-s256_l512-yuv420-val'}",
    ]


def test_path_quiet_and_dest(env, capsys):
    assert cli.main(["path", "in1k", "val", "--fmt", "yuv420", "-q", "--dest", "/x"]) == 0
    assert capsys.readouterr().out.strip() == "/x/imagenet1k-s256_l512-yuv420-val"


def test_path_no_such_combo(env):
    with pytest.raises(SystemExit) as ei:
        cli.main(["path", "imagenette", "val"])
    assert "no cache" in str(ei.value)


# --------------------------------------------------------------------------- #
# sync
# --------------------------------------------------------------------------- #


def test_sync_dry_run_expands_splits_and_fmts(env, capsys):
    make_cache(env.root, "imagenet100-s256_l512-jpeg-val")
    rc = cli.main(["sync", "imagenet100", "train,val", "all", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "✓ imagenet100-s256_l512-jpeg-val: already present" in out
    assert sorted(env.s3fake.listed) == [
        "s3://visionlab-datasets/slipstream-cache/imagenet100/imagenet100-s256_l512-jpeg-train/",
        "s3://visionlab-datasets/slipstream-cache/imagenet100/imagenet100-s256_l512-yuv420-train/",
        "s3://visionlab-datasets/slipstream-cache/imagenet100/imagenet100-s256_l512-yuv420-val/",
    ]
    assert "[dry-run] nothing downloaded" in out
    assert env.downloads == []


IN10_VAL = "imagenet10-s256_l512-jpeg-val"
IN10_VAL_REMOTE = f"s3://visionlab-datasets/slipstream-cache/imagenet10/{IN10_VAL}/"


def test_sync_downloads_and_verifies(env, capsys):
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    out = capsys.readouterr().out
    assert rc == 0, out
    target = env.root / IN10_VAL
    staging = env.root / f".{IN10_VAL}.sync.partial"
    # manifest first (own fetch), then the data files, all into the staging dir
    assert [(r, d, n) for r, d, n in env.downloads] == [
        (IN10_VAL_REMOTE, staging, [MANIFEST]),
        (IN10_VAL_REMOTE, staging, ["image.bin", "image.meta.npy", "label.npy"]),
    ]
    assert f"✓ {IN10_VAL} -> {target}" in out
    assert sorted(p.name for p in target.iterdir()) == ["image.bin", "image.meta.npy", "label.npy", MANIFEST]
    assert not staging.exists() and not (env.root / f".{IN10_VAL}.sync.lock").exists()


def test_sync_requires_splits_and_fmt(env, capsys):
    for argv in (["sync", "imagenet10"], ["sync", "imagenet10", "val"]):
        with pytest.raises(SystemExit) as ei:
            cli.main(argv)
        assert ei.value.code == 2
    assert "required: splits, fmt" in capsys.readouterr().err
    assert env.downloads == []


def test_sync_accepts_alias(env):
    cli.main(["sync", "in10", "val", "jpeg"])
    assert {d.name for _, d, _ in env.downloads} == {".imagenet10-s256_l512-jpeg-val.sync.partial"}


def test_sync_skips_unregistered_combo_with_warning(env, capsys):
    from visionlab.datasets import get_config, register
    from visionlab.datasets.registry import DatasetConfig, REGISTRY

    cfg = get_config("imagenet10")
    partial = DatasetConfig(
        name="partial10",
        num_classes=10,
        remote_cache={("val", "jpeg"): cfg.remote_cache[("val", "jpeg")]},
        metadata={},
    )
    register(partial)
    try:
        rc = cli.main(["sync", "partial10", "train,val", "jpeg", "--dry-run"])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "⚠ partial10: no train/jpeg cache registered, skipping" in out
    finally:
        REGISTRY.pop("partial10", None)


def test_sync_remote_denied(env, capsys):
    env.s3fake.error = Exception("AccessDenied: nope")
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "✗ imagenet10-s256_l512-jpeg-val: remote denied" in out
    assert env.downloads == []


def test_sync_unknown_dataset(env):
    with pytest.raises(SystemExit):
        cli.main(["sync", "cifar", "val", "jpeg"])
    assert env.downloads == []


def test_sync_dataset_without_caches(env):
    with pytest.raises(SystemExit) as ei:
        cli.main(["sync", "imagenette", "val", "jpeg"])
    assert "no cache" in str(ei.value)


# --------------------------------------------------------------------------- #
# slipstream missing
# --------------------------------------------------------------------------- #


def test_missing_slipstream_cli_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "slipstream.cli":
            raise ImportError("no cli")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as ei:
        cli._slipstream_cli()
    assert "uv lock --upgrade-package visionlab-slipstream" in str(ei.value)


def test_main_dispatch_returns_int(env):
    ns = argparse.Namespace(json=True)
    assert cli.cmd_list(ns) == 0


# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #


def test_colorize_paints_glyphs_only_when_enabled(monkeypatch):
    line = "  imagenet10  ✓   34.9 MB   ✗ missing   ⚠ x"
    monkeypatch.setattr(cli, "_COLOR", False)
    assert cli.colorize(line) == line
    monkeypatch.setattr(cli, "_COLOR", True)
    out = cli.colorize(line)
    assert "\033[32m✓\033[0m" in out and "\033[31m✗\033[0m" in out and "\033[33m⚠\033[0m" in out
    # padding is computed before colouring, so stripping codes gives the original line
    import re

    assert re.sub(r"\033\[[0-9;]*m", "", out) == line


def test_configure_color_modes(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    cli.configure_color("auto")
    assert cli._COLOR is False  # NO_COLOR wins in auto mode
    cli.configure_color("always")
    assert cli._COLOR is True  # explicit flag wins over NO_COLOR
    cli.configure_color("never")
    assert cli._COLOR is False
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    cli.configure_color("auto")
    assert cli._COLOR is True


def test_no_color_flag_and_piped_output_plain(env, capsys):
    make_cache(env.root, "imagenet10-s256_l512-jpeg-val")
    cli.main(["--no-color", "path", "in10", "val"])
    assert "\033[" not in capsys.readouterr().out
    cli.main(["path", "in10", "val"])  # capsys is not a TTY -> auto = off
    assert "\033[" not in capsys.readouterr().out
    cli.main(["--color", "always", "path", "in10", "val"])
    assert "\033[32m✓\033[0m imagenet10 val jpeg" in capsys.readouterr().out


def test_platform_dirs_are_unexpanded_and_not_printed(env, capsys, monkeypatch):
    from visionlab.datasets.runtime_platform import PLATFORM_CACHE_DIRS, Platform, get_platform_cache_dir

    assert PLATFORM_CACHE_DIRS[Platform.CPU_WORKSTATION] == "~/.slipstream"
    with monkeypatch.context() as m:
        m.delenv("SLIPSTREAM_CACHE_DIR")
        assert get_platform_cache_dir(Platform.CPU_WORKSTATION) == str(Path.home() / ".slipstream")
    cli.main(["status", "--no-remote"])
    out = capsys.readouterr().out
    assert "platforms" not in out
    assert "Cache directory  (slipstream caches on this machine live here)" in out


def test_status_reports_culled_dir_as_empty(env, capsys):
    (env.root / "imagenet10-s256_l512-jpeg-val").mkdir()  # dir survives a cull, files don't
    rc = cli.main(["status", "--no-remote"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "imagenet10        val    jpeg    ✗ empty dir" in out
    assert "1 cache dir(s) exist without manifest.json" in out


def test_sync_redownloads_empty_dir(env, capsys):
    (env.root / "imagenet10-s256_l512-jpeg-val").mkdir()
    rc = cli.main(["sync", "imagenet10", "val", "jpeg", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "dir exists but no manifest (culled?)" in out


def test_fas_cluster_default_is_persistent_storage():
    from visionlab.datasets.runtime_platform import PLATFORM_CACHE_DIRS, Platform

    assert PLATFORM_CACHE_DIRS[Platform.FAS_CLUSTER] == "/n/lab_storage/alvarez_lab/Lab/datasets/slipstream"
    assert "netscratch" not in PLATFORM_CACHE_DIRS[Platform.FAS_CLUSTER]


def test_sync_passes_s5cmd_tuning(env, monkeypatch):
    seen = {}
    real = env.s3fake.fetch

    def fetch(remote, names, dest, **kw):
        if names != [MANIFEST]:
            seen.update({k: kw[k] for k in ("concurrency", "part_size_mb", "numworkers")})
        return real(remote, names, dest, **kw)

    monkeypatch.setattr(cli.cache_sync, "fetch_files", fetch)
    assert cli.main(["sync", "imagenet10", "val", "jpeg", "--part-size", "128", "--numworkers", "8"]) == 0
    assert seen == {"concurrency": 1, "part_size_mb": 128, "numworkers": 8}


def test_status_detects_inflight_download(env, capsys, monkeypatch):
    from slipstream.cache import OptimizedCache

    d = make_cache(env.root, "imagenet100-s256_l512-yuv420-train")
    (d / "image.bin3335549746").write_bytes(b"x" * 2048)  # s5cmd temp file
    monkeypatch.setattr(OptimizedCache, "check_integrity", staticmethod(lambda p: (False, ["missing: image.bin"])))
    cli.main(["status", "--no-remote"])
    out = capsys.readouterr().out
    assert "imagenet100       train  yuv420  ⚠ downloading" in out
    assert "download in progress: image.bin (2.0 KB so far)" in out
    assert "missing: image.bin" not in out


# --------------------------------------------------------------------------- #
# sync: shared-dir safety (lock, staging, verify, repair)
# --------------------------------------------------------------------------- #


def _synced(env) -> Path:
    assert cli.main(["sync", "imagenet10", "val", "jpeg"]) == 0
    env.downloads.clear()
    return env.root / IN10_VAL


def test_sync_repairs_only_missing_files(env, capsys):
    target = _synced(env)
    (target / "image.bin").unlink()  # scratch purge: a data file and the manifest gone
    (target / MANIFEST).unlink()
    (target / "label.npy").write_bytes(b"short")  # truncated
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert [n for _, _, n in env.downloads] == [[MANIFEST], ["image.bin", "label.npy"]]
    assert "fetching 2 of 3 data files" in out and "1 already correct" in out
    assert (target / "label.npy").read_bytes() == b"l" * 50
    assert (target / MANIFEST).exists()


def test_sync_present_cache_skipped_unless_deep_or_force(env, capsys):
    _synced(env)
    assert cli.main(["sync", "imagenet10", "val", "jpeg"]) == 0
    assert "already present" in capsys.readouterr().out
    assert env.downloads == []


def test_sync_refuses_while_another_sync_holds_the_lock(env, capsys):
    import os
    import socket

    lock = env.root / f".{IN10_VAL}.sync.lock"
    lock.write_text(json.dumps({"user": "someone", "host": socket.gethostname(), "pid": os.getpid(),
                                "started": 0, "token": "t"}))
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "another sync holds" in out and "someone@" in out
    assert env.downloads == [] and not (env.root / IN10_VAL).exists()
    assert lock.exists()  # not ours: left alone


def test_sync_breaks_stale_lock(env, capsys):
    import socket

    lock = env.root / f".{IN10_VAL}.sync.lock"
    lock.write_text(json.dumps({"user": "ghost", "host": socket.gethostname(), "pid": 2**22 + 12345,
                                "started": 0, "token": "t"}))
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "broke stale sync lock" in out and "is gone" in out
    assert not lock.exists()


def test_stale_lock_by_heartbeat_age(tmp_path):
    import os
    import time

    target = tmp_path / "c"
    lp = cli.cache_sync.lock_path(target)
    lp.write_text(json.dumps({"user": "u", "host": "other-host", "pid": 1, "started": 0, "token": "t"}))
    info = cli.cache_sync.read_lock(target)
    assert cli.cache_sync.lock_stale_reason(info, lp) is None  # other host, fresh heartbeat: live
    old = time.time() - cli.cache_sync.STALE_S - 5
    os.utime(lp, (old, old))
    info = cli.cache_sync.read_lock(target)
    assert "no heartbeat" in cli.cache_sync.lock_stale_reason(info, lp)


def test_sync_failure_keeps_good_copy_and_staging(env, capsys):
    target = _synced(env)
    (target / "image.bin").unlink()
    (target / "label.npy").unlink()
    env.s3fake.fail_after = 1  # fetches image.bin, then "fails"
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    out = capsys.readouterr().out
    assert rc == 1
    staging = env.root / f".{IN10_VAL}.sync.partial"
    assert "staged files kept" in out
    marker = json.loads((staging / "SYNC_INCOMPLETE.json").read_text())
    assert "s5cmd reported errors" in marker["reason"]
    assert (target / "image.meta.npy").exists() and (target / MANIFEST).exists()  # untouched
    assert not (target / "image.bin").exists()  # nothing half-committed
    # status reports the unfinished sync; the rerun resumes from staging
    cli.main(["status", "--no-remote"])
    assert "unfinished sync staged in" in capsys.readouterr().out
    env.s3fake.fail_after = None
    env.downloads.clear()
    assert cli.main(["sync", "imagenet10", "val", "jpeg"]) == 0
    assert [n for _, _, n in env.downloads] == [[MANIFEST], ["label.npy"]]  # image.bin reused
    assert (target / "image.bin").exists() and not staging.exists()


def test_sync_refuses_to_mix_cache_versions(env, capsys):
    target = _synced(env)
    (target / "label.npy").unlink()
    objs = env.s3fake.objects[IN10_VAL_REMOTE]
    objs[MANIFEST] = fake_manifest({n: b for n, b in objs.items() if n != MANIFEST}, rebuilt=True)
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    assert rc == 1
    assert "differs from the local one" in capsys.readouterr().out
    assert "rebuilt" not in json.loads((target / MANIFEST).read_text())


def _hashed_remote(env) -> dict[str, bytes]:
    import hashlib

    objs = env.s3fake.remote(IN10_VAL_REMOTE)
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in objs.items() if n != MANIFEST}
    objs[MANIFEST] = fake_manifest({n: b for n, b in objs.items() if n != MANIFEST}, file_sha256=hashes)
    return objs


def test_sync_rejects_corrupt_download_by_sha256(env, capsys):
    _hashed_remote(env)
    env.s3fake.corrupt = {"image.bin"}
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "sha256 mismatch: image.bin" in out
    staging = env.root / f".{IN10_VAL}.sync.partial"
    assert not (staging / "image.bin").exists() and (staging / "label.npy").exists()
    assert not (env.root / IN10_VAL).exists()


def test_status_deep_and_sync_deep_repair(env, capsys):
    _hashed_remote(env)
    target = _synced(env)
    capsys.readouterr()
    assert cli.main(["status", "--no-remote", "--deep", "--json"]) == 0
    row = next(d for d in json.loads(capsys.readouterr().out)["datasets"] if d["cache_name"] == IN10_VAL)
    assert row["deep_status"] == "ok"

    (target / "image.bin").write_bytes(b"Z" * 3000)  # same size, bad content: sizes can't see it
    assert cli.main(["status", "--no-remote", "--deep"]) == 1
    out = capsys.readouterr().out
    assert "1 mismatch" in out and "sha256 mismatch: image.bin" in out

    assert cli.main(["sync", "imagenet10", "val", "jpeg", "--deep"]) == 0
    out = capsys.readouterr().out
    assert "sha256 mismatch in cache: image.bin (will re-fetch)" in out
    assert env.downloads[-1][2] == ["image.bin"]
    assert (target / "image.bin").read_bytes() == b"i" * 3000


def test_status_deep_without_hashes_is_unavailable(env, capsys):
    _synced(env)
    capsys.readouterr()
    assert cli.main(["status", "--no-remote", "--deep", "--json"]) == 0
    row = next(d for d in json.loads(capsys.readouterr().out)["datasets"] if d["cache_name"] == IN10_VAL)
    assert row["deep_status"] == "unavailable"


def test_status_shows_live_sync(env, capsys):
    import os
    import socket

    lock = env.root / f".{IN10_VAL}.sync.lock"
    lock.write_text(json.dumps({"user": "alice", "host": socket.gethostname(), "pid": os.getpid(),
                                "started": 0, "token": "t"}))
    cli.main(["status", "--no-remote", "--json"])
    row = next(d for d in json.loads(capsys.readouterr().out)["datasets"] if d["cache_name"] == IN10_VAL)
    assert row["local_status"] == "syncing" and row["sync_lock"]["user"] == "alice"
    assert row["sync_lock"]["stale"] is None


def test_shared_dir_files_group_writable(env, capsys):
    import os
    import stat

    os.chmod(env.root, 0o2775)
    target = _synced(env)
    for p in target.iterdir():
        assert p.stat().st_mode & stat.S_IWGRP, p


def test_ensure_cache_syncs_then_is_idempotent(env):
    target = env.root / IN10_VAL
    res = cli.cache_sync.ensure_cache(IN10_VAL_REMOTE, target, log=lambda m: None)
    assert res.ok and (target / MANIFEST).exists() and res.fetched_files == 3


def test_ensure_cache_waits_for_another_sync(env):
    import os
    import socket
    import threading
    import time

    target = env.root / IN10_VAL
    lock = cli.cache_sync.lock_path(target)
    lock.write_text(json.dumps({"user": "bob", "host": socket.gethostname(), "pid": os.getpid(),
                                "started": 0, "token": "t"}))

    def other_sync_finishes():
        time.sleep(0.3)
        objs = env.s3fake.remote(IN10_VAL_REMOTE)
        target.mkdir()
        for n, b in objs.items():
            (target / n).write_bytes(b)
        lock.unlink()

    threading.Thread(target=other_sync_finishes).start()
    logs: list[str] = []
    res = cli.cache_sync.ensure_cache(IN10_VAL_REMOTE, target, poll_s=0.05, log=logs.append)
    assert res.ok
    assert any("waiting for another sync" in m and "bob@" in m for m in logs)
    assert env.downloads == []  # used bob's copy, no second download


def test_ensure_cache_raises_on_failure(env):
    env.s3fake.fail_after = 0
    with pytest.raises(RuntimeError, match="s5cmd reported errors"):
        cli.cache_sync.ensure_cache(IN10_VAL_REMOTE, env.root / IN10_VAL, log=lambda m: None)
