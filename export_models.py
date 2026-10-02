"""Export only trained models and aggregate evaluation files for publication.

Run after prepare_models.py, in the same Python 3.12 environment.
The original Numerai-uploaded artifacts are left unchanged.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import types

import cloudpickle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NAMES = ("hello_numerai", "feature_neutralization", "target_ensemble")


def portable_code(code):
    constants = tuple(portable_code(v) if isinstance(v, types.CodeType) else v
                      for v in code.co_consts)
    return code.replace(co_filename="prepare_models.py", co_consts=constants)


def main():
    source = ROOT / "artifacts"
    models = ROOT / "models"
    reports = ROOT / "reports"
    models.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    training = json.loads((source / "manifest.json").read_text())
    live = pd.read_parquet(ROOT / "data" / f"round-{training['live_round']}-live.parquet")
    files = {}
    for name in NAMES:
        original = cloudpickle.loads((source / f"{name}.pkl").read_bytes())
        namespace = dict(original.__globals__)
        namespace.update(__file__="prepare_models.py", __name__="__main__", __package__=None)
        portable = types.FunctionType(portable_code(original.__code__), namespace,
                                      original.__name__, original.__defaults__, original.__closure__)
        portable.__kwdefaults__ = original.__kwdefaults__
        payload = cloudpickle.dumps(portable)
        # Absolute source paths are unnecessary for inference and are omitted.
        assert str(ROOT).encode() not in payload
        filename = f"{name}.pkl"
        (models / filename).write_bytes(payload)
        published = cloudpickle.loads(payload)
        expected = original(live, None)
        actual = published(live, None)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert np.isfinite(actual.values).all()
        files[filename] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        print(f"{filename}: exact prediction match on {len(actual):,} rows")
    for filename in ("validation_summary.json", "validation_per_era.csv"):
        shutil.copyfile(source / filename, reports / filename)
    manifest = {
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "cloudpickle", "lightgbm", "numpy", "pandas", "scipy", "scikit-learn")},
        "data_version": training["data_version"],
        "targets": training["targets"],
        "training_rows": training["training_rows"],
        "training_eras": training["training_eras"],
        "feature_count": len(training["features"]),
        "parameters": training["parameters"],
        "publication_check": {"live_round": training["live_round"], "rows": len(live),
                              "exact_prediction_match": True},
        "files": files,
    }
    (models / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
