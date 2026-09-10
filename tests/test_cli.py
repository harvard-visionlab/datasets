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
    monkeypatch.setattr(OptimizedCache, "check_integrity", staticmethod(lambda p: (True, [])))
    downloads: list[tuple[str, Path]] = []

    def fake_download(remote, local, endpoint_url=None, numworkers=32, verbose=True):
        downloads.append((remote, Path(local)))
        make_cache(Path(local).parent, Path(local).name, num_samples=99)
        return True

    import slipstream.s3_sync

    monkeypatch.setattr(slipstream.s3_sync, "download_s3_cache", fake_download)
    plumbing.downloads = downloads
    plumbing.root = tmp_path
    return plumbing


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
    assert sorted(env.listed) == [
        "s3://visionlab-datasets/slipstream-cache/imagenet100/imagenet100-s256_l512-jpeg-train/",
        "s3://visionlab-datasets/slipstream-cache/imagenet100/imagenet100-s256_l512-yuv420-train/",
        "s3://visionlab-datasets/slipstream-cache/imagenet100/imagenet100-s256_l512-yuv420-val/",
    ]
    assert "[dry-run] nothing downloaded" in out
    assert env.downloads == []


def test_sync_downloads_and_verifies(env, capsys):
    rc = cli.main(["sync", "imagenet10", "val", "jpeg"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert env.downloads == [
        (
            "s3://visionlab-datasets/slipstream-cache/imagenet10/imagenet10-s256_l512-jpeg-val/",
            env.root / "imagenet10-s256_l512-jpeg-val",
        )
    ]
    assert f"✓ imagenet10-s256_l512-jpeg-val -> {env.root / 'imagenet10-s256_l512-jpeg-val'}" in out


def test_sync_requires_splits_and_fmt(env, capsys):
    for argv in (["sync", "imagenet10"], ["sync", "imagenet10", "val"]):
        with pytest.raises(SystemExit) as ei:
            cli.main(argv)
        assert ei.value.code == 2
    assert "required: splits, fmt" in capsys.readouterr().err
    assert env.downloads == []


def test_sync_accepts_alias(env):
    cli.main(["sync", "in10", "val", "jpeg"])
    assert [p.name for _, p in env.downloads] == ["imagenet10-s256_l512-jpeg-val"]


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
    env.remote_error = Exception("AccessDenied: nope")
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
