"""Tests for the ``visionlab-datasets`` CLI (offline; slipstream.cli is patched)."""
import argparse
import json
from pathlib import Path

import pytest

import visionlab.datasets.cli as cli


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

    known = set(list_datasets())
    assert set(cli.ALIASES.values()) <= known


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


# --------------------------------------------------------------------------- #
# commands (slipstream.cli patched)
# --------------------------------------------------------------------------- #


class FakeSlipCli:
    OK, BAD, WARN = "✓", "✗", "⚠"
    MANIFEST_FILE = "manifest.json"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.sync_calls: list[argparse.Namespace] = []
        self.printed = None

    def _import_registry(self):
        import visionlab.datasets as vd

        return vd

    def _configure_color(self, no_color=False):
        pass

    def resolve_cache_dir(self, vd):
        return argparse.Namespace(path=str(self.cache_dir))

    def cmd_sync(self, ns):
        self.sync_calls.append(ns)
        return 0

    def collect_status(self, check_remote_access=True, endpoint_url=None):
        return {
            "visionlab_datasets_version": "x",
            "cache": {"path": str(self.cache_dir), "access": {}},
            "datasets": [
                {
                    "dataset": "imagenet10",
                    "split": "val",
                    "fmt": "jpeg",
                    "local_status": "ok",
                    "local_path": str(self.cache_dir / "imagenet10-s256_l512-jpeg-val"),
                },
                {
                    "dataset": "imagenet10",
                    "split": "train",
                    "fmt": "jpeg",
                    "local_status": "missing",
                    "local_path": str(self.cache_dir / "imagenet10-s256_l512-jpeg-train"),
                },
            ],
        }

    def print_status(self, status):
        self.printed = status
        print("STATUS TABLE")

    def _problems(self, status):
        return []


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeSlipCli(tmp_path)
    monkeypatch.setattr(cli, "_slipstream_cli", lambda: f)
    return f


def test_sync_expands_alias_splits_and_fmts(fake):
    rc = cli.main(["sync", "in100", "train,val", "--fmt", "all", "--dry-run", "--numworkers", "4"])
    assert rc == 0
    calls = [(ns.targets, ns.split, ns.fmt) for ns in fake.sync_calls]
    assert calls == [
        (["imagenet100"], "train", "jpeg"),
        (["imagenet100"], "train", "yuv420"),
        (["imagenet100"], "val", "jpeg"),
        (["imagenet100"], "val", "yuv420"),
    ]
    assert all(ns.dry_run and ns.numworkers == 4 and ns.dest is None for ns in fake.sync_calls)


def test_sync_defaults_to_val_jpeg(fake):
    assert cli.main(["sync", "in10"]) == 0
    assert [(ns.split, ns.fmt) for ns in fake.sync_calls] == [("val", "jpeg")]


def test_sync_unknown_dataset(fake):
    with pytest.raises(SystemExit):
        cli.main(["sync", "cifar"])
    assert fake.sync_calls == []


def test_sync_dataset_without_caches(fake):
    with pytest.raises(SystemExit) as ei:
        cli.main(["sync", "imagenette", "val"])
    assert "no cache" in str(ei.value)


def test_path_prints_local_paths(fake, capsys, tmp_path):
    (tmp_path / "imagenet10-s256_l512-jpeg-val").mkdir()
    (tmp_path / "imagenet10-s256_l512-jpeg-val" / "manifest.json").write_text("{}")
    assert cli.main(["path", "in10", "all"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"✗ imagenet10 train jpeg  {tmp_path / 'imagenet10-s256_l512-jpeg-train'}",
        f"✓ imagenet10 val jpeg  {tmp_path / 'imagenet10-s256_l512-jpeg-val'}",
    ]


def test_path_quiet_and_dest(fake, capsys):
    assert cli.main(["path", "in1k", "val", "--fmt", "yuv420", "-q", "--dest", "/x"]) == 0
    assert capsys.readouterr().out.strip() == "/x/imagenet1k-s256_l512-yuv420-val"


def test_status_text_with_paths(fake, capsys):
    assert cli.main(["status", "--paths", "--no-remote"]) == 0
    out = capsys.readouterr().out
    assert "STATUS TABLE" in out
    assert "Local cache paths" in out
    assert "imagenet10-s256_l512-jpeg-val" in out
    assert "imagenet10-s256_l512-jpeg-train" not in out  # missing entries not listed
    assert "in100=imagenet100" in out
    assert fake.printed["visionlab_datasets_version"] == cli.__version__


def test_status_json(fake, capsys):
    assert cli.main(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["aliases"]["in1k"] == "imagenet1k"
    assert data["visionlab_datasets_version"] == cli.__version__


def test_list_text(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "imagenet100  (100 classes)  [in100]" in out
    assert "imagenette  (10 classes)" in out
    assert "(no remote caches registered)" in out
    assert "s3://visionlab-datasets/slipstream-cache/imagenet100/imagenet100-s292_l584-jpeg-val" in out


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
