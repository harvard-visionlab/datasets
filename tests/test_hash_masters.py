"""hash_masters: pure helpers (S3/local hashing is exercised on real data, not here)."""
import json

from visionlab.datasets.prep import hash_masters as H

MAN = {"version": 1, "num_samples": 2, "fields": {"image": {"type": "ImageBytes"}, "label": {"type": "int"}},
       "file_sizes": {"image.bin": 10, "image.meta.npy": 2, "label.npy": 3}}
HASHES = {"image.bin": "a" * 64, "image.meta.npy": "b" * 64, "label.npy": "c" * 64}


def entry(**kw):
    return {"manifest": MAN, "manifest_sha256": "x", "file_sizes": MAN["file_sizes"], "file_sha256": HASHES, **kw}


def test_data_files_match_file_sizes():
    assert sorted(H.data_files(MAN)) == sorted(MAN["file_sizes"])


def test_hashed_manifest_adds_only_hashes():
    m = H.hashed_manifest(MAN, HASHES)
    assert m["file_sha256"] == HASHES and {k: v for k, v in m.items() if k != "file_sha256"} == MAN
    assert H.serialise(m) == json.dumps(m, indent=2).encode()


def test_compare():
    assert H.compare({"c": entry()}, {"c": entry()}) == {"c": []}
    hashed_side = entry(manifest={**MAN, "file_sha256": HASHES})  # a copy already hashed: same build
    assert H.compare({"c": entry()}, {"c": hashed_side}) == {"c": []}
    bad = entry(file_sha256={**HASHES, "label.npy": "d" * 64})
    assert H.compare({"c": entry()}, {"c": bad}) == {"c": ["sha256 differs: label.npy"]}
    other = entry(manifest={**MAN, "num_samples": 3})
    assert H.compare({"c": entry()}, {"c": other})["c"] == ["manifests describe different builds"]
