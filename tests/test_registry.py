

def test_cache_identity(tmp_path):
    import json

    from visionlab.datasets import cache_identity

    m = {"version": 3, "num_samples": 2, "fields": {}, "file_sha256": {"a.bin": "11", "b.bin": "22"}}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    ident = cache_identity(tmp_path)
    assert ident.startswith("sha256:") and len(ident) == 7 + 64
    (tmp_path / "manifest.json").write_text(json.dumps(dict(reversed(list(m.items()))), indent=4))
    assert cache_identity(tmp_path) == ident  # formatting / key order don't matter
    m["file_sha256"]["b.bin"] = "23"
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    assert cache_identity(tmp_path) != ident
    del m["file_sha256"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    assert cache_identity(tmp_path) is None


def test_imagenet_original_registered():
    from visionlab.datasets import get_config

    for name, cache in (("imagenet1k", "imagenet1k-original-val"), ("imagenet100", "imagenet100-original-val")):
        assert get_config(name).remote_cache[("val", "original")].endswith(f"/{name}/{cache}")
