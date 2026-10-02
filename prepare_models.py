"""Reproducible local adaptation of Numerai's three onboarding notebooks.

Uses Numerai-provided data locally; datasets are not redistributed.
Does not create keys, upload, stake, or change an account.
Run with Python 3.12 and the pinned requirements.txt.
"""
from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import cloudpickle
import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_file
from numerapi import NumerAPI

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "artifacts"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
VERSION = "v5.3"
TARGETS = ["target_ender_60", "target_teager2b_60"]
NAPI = NumerAPI(verbosity="warning", show_progress_bars=False)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def project_dataset(name, columns, round_num=None):
    """Download required Parquet columns in bounded row-group batches."""
    dest = DATA / (f"round-{round_num}-{name}" if round_num is not None else name)
    if dest.exists():
        assert set(columns) <= set(pq.ParquetFile(dest).schema.names)
        return dest
    url = NAPI.raw_query(
        "query($filename: String!, $round: Int) { dataset(filename: $filename, round: $round) }",
        {"filename": f"{VERSION}/{name}", "round": round_num},
    )["data"]["dataset"]
    with fsspec.open(url, "rb", block_size=256 * 1024) as remote:
        metadata = pq.ParquetFile(remote).metadata
    log(f"Downloading {name}: {metadata.num_rows:,} rows; {len(columns)} columns")
    partial = dest.with_suffix(".partial")
    writer = None
    try:
        for group in range(metadata.num_row_groups):
            with open_parquet_file(
                url, columns=columns, row_groups=[group], engine="pyarrow",
                max_gap=1024, max_block=32 * 1024 * 1024,
                footer_sample_size=4 * 1024 * 1024,
            ) as remote:
                table = pq.ParquetFile(remote).read_row_group(group, columns=columns)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
            writer.write_table(table)
            del table
            gc.collect()
            log(f"{name} row group {group + 1}/{metadata.num_row_groups}")
    finally:
        if writer is not None:
            writer.close()
    partial.replace(dest)
    return dest


def make_predictors(base, other, feature_names, neutral_names):
    """Return self-contained callables suitable for cloudpickle uploads."""
    def baseline(live_features, live_benchmark_models=None):
        raw = base.predict(live_features[feature_names], num_threads=1)
        assert np.isfinite(raw).all(), "Invalid raw model output"
        return pd.DataFrame({"prediction": raw}, index=live_features.index).rank(pct=True)

    def neutral(live_features, live_benchmark_models=None):
        raw = base.predict(live_features[feature_names], num_threads=1)
        exposures = live_features[neutral_names].to_numpy(dtype=np.float64)
        exposures = np.column_stack([exposures, np.ones(len(exposures))])
        assert np.isfinite(raw).all() and np.isfinite(exposures).all()
        # Same full linear neutralization as the official agility example.
        coefficients = np.linalg.lstsq(exposures, raw, rcond=1e-6)[0]
        adjustments = np.einsum("ij,j->i", exposures, coefficients)
        values = raw - adjustments
        assert np.isfinite(values).all(), "Invalid neutralized model output"
        return pd.DataFrame({"prediction": values}, index=live_features.index).rank(pct=True)

    def ensemble(live_features, live_benchmark_models=None):
        x = live_features[feature_names]
        predictions = pd.DataFrame({
            "ender": base.predict(x, num_threads=1),
            "teager": other.predict(x, num_threads=1),
        }, index=live_features.index)
        assert np.isfinite(predictions.to_numpy()).all(), "Invalid ensemble model output"
        average = predictions.rank(pct=True).mean(axis=1)
        return average.rank(pct=True).to_frame("prediction")

    return {"hello_numerai": baseline, "feature_neutralization": neutral,
            "target_ensemble": ensemble}


def validate_predictions(predictions, index):
    assert index.is_unique, "Duplicate stock IDs"
    assert predictions.index.equals(index), "IDs or row ordering changed"
    assert predictions.columns.tolist() == ["prediction"]
    values = predictions["prediction"]
    assert np.isfinite(values).all(), "Non-finite predictions"
    assert values.between(0, 1).all(), "Predictions outside [0, 1]"
    assert values.nunique() > 1, "Constant predictions"


def main():
    import lightgbm as lgb
    from numerai_tools.scoring import numerai_corr
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=4)
    meta_path = DATA / "features.json"
    if not meta_path.exists():
        NAPI.download_dataset(f"{VERSION}/features.json", str(meta_path))
    metadata = json.loads(meta_path.read_text())
    features = metadata["feature_sets"]["faith"]
    neutral_features = sorted(set(metadata["feature_sets"]["medium"]) &
                              set(metadata["feature_sets"]["agility"]))
    train_path = project_dataset("train.parquet", ["id", "era"] + TARGETS + features)
    train = pd.read_parquet(train_path)
    # Constant training features cannot contribute to either tree model.
    features = [name for name in features if train[name].nunique() > 1]
    last_train_era = int(train.era.max())
    manifest = {
        "python": platform.python_version(), "data_version": VERSION,
        "targets": TARGETS, "training_rows": len(train),
        "training_eras": int(train.era.nunique()),
        "last_training_era": last_train_era, "embargo_eras": 12,
        "features": features, "neutral_features": neutral_features,
        "sources": [f"https://github.com/numerai/example-scripts/blob/master/numerai/{n}.ipynb"
                    for n in ["hello_numerai", "feature_neutralization", "target_ensemble"]],
    }
    x = train[features]
    models = {}
    parameters = dict(n_estimators=2000, learning_rate=0.01, max_depth=5,
                      num_leaves=31, colsample_bytree=0.1, n_jobs=4,
                      random_state=42, verbosity=-1, force_col_wise=True)
    manifest["parameters"] = parameters
    for target in TARGETS:
        model_path = OUT / f"{target}.txt"
        if model_path.exists():
            models[target] = lgb.Booster(model_file=str(model_path))
        else:
            valid = train[target].notna()
            log(f"Training {target}: {valid.sum():,} rows, {len(features)} features")
            estimator = lgb.LGBMRegressor(**parameters)
            estimator.fit(x.loc[valid], train.loc[valid, target])
            estimator.booster_.save_model(str(model_path))
            models[target] = estimator.booster_
            del estimator
            gc.collect()
            log(f"Saved {target}")
    del train, x
    gc.collect()

    predictors = make_predictors(models[TARGETS[0]], models[TARGETS[1]],
                                 features, neutral_features)
    for name, predictor in predictors.items():
        (OUT / f"{name}.pkl").write_bytes(cloudpickle.dumps(predictor))

    required_features = list(dict.fromkeys(features + neutral_features))
    live_round = NAPI.get_current_round()
    assert live_round is not None, "Unable to identify the current tournament round"
    live_path = project_dataset("live.parquet", ["id"] + required_features, live_round)
    live = pd.read_parquet(live_path)
    manifest["live_rows"] = len(live)
    manifest["live_round"] = live_round
    for name in predictors:
        # Reload our own artifacts to test the actual serialized callable.
        predictor = cloudpickle.loads((OUT / f"{name}.pkl").read_bytes())
        start = time.monotonic()
        result = predictor(live, None)
        validate_predictions(result, live.index)
        result.to_csv(OUT / f"{name}_live.csv")
        log(f"{name}: live smoke test passed ({len(result):,} rows, {time.monotonic() - start:.1f}s)")
    del live
    gc.collect()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Streaming validation uses every eligible era, never the test rows.
    validation_path = project_dataset("validation.parquet",
                                     ["id", "era", "data_type", "target"] + required_features)
    parquet = pq.ParquetFile(validation_path)
    pending = pd.DataFrame()
    scores = []

    class PredictionCache:
        """Reuse real batched tree predictions across the three pipelines."""
        def __init__(self, values):
            self.values = values

        def predict(self, frame, num_threads=1):
            return self.values.loc[frame.index].to_numpy()

    def evaluate_era(era, frame):
        if int(era) <= last_train_era + 12:
            return
        frame = frame.loc[(frame.data_type == "validation") & frame.target.notna()]
        if len(frame) < 2:
            return
        score = {"era": str(era), "rows": len(frame)}
        cached_predictors = make_predictors(
            PredictionCache(frame["_raw_base"]), PredictionCache(frame["_raw_other"]),
            features, neutral_features,
        )
        for name, predictor in cached_predictors.items():
            result = predictor(frame, None)
            validate_predictions(result, frame.index)
            if not scores:
                deployed_result = predictors[name](frame, None)
                assert np.allclose(result, deployed_result, rtol=0, atol=1e-12)
            score[name] = float(numerai_corr(result, frame.target).iloc[0])
        scores.append(score)
        if len(scores) % 50 == 0:
            log(f"Validated {len(scores)} eras")

    for batch in parquet.iter_batches(batch_size=50_000):
        frame = batch.to_pandas()
        frame = frame.loc[(frame.data_type == "validation") & frame.target.notna() &
                          (frame.era.astype(int) > last_train_era + 12)].copy()
        if frame.empty:
            continue
        frame["_raw_base"] = models[TARGETS[0]].predict(frame[features], num_threads=4)
        frame["_raw_other"] = models[TARGETS[1]].predict(frame[features], num_threads=4)
        frame = pd.concat([pending, frame])
        last_era = frame.era.iloc[-1]
        for era, era_frame in frame.loc[frame.era != last_era].groupby("era", sort=True):
            evaluate_era(era, era_frame)
        pending = frame.loc[frame.era == last_era].copy()
    if len(pending):
        evaluate_era(pending.era.iloc[0], pending)
    per_era = pd.DataFrame(scores).set_index("era").sort_index()
    assert per_era.index.is_unique, "Validation input eras must be contiguous"
    per_era.to_csv(OUT / "validation_per_era.csv")
    summary = {}
    for name in predictors:
        corr = per_era[name]
        cumulative = np.r_[0, corr.cumsum().to_numpy()]
        std = float(corr.std(ddof=0))
        summary[name] = {
            "validation_eras": len(corr), "corr_mean": float(corr.mean()),
            "corr_std": std, "corr_sharpe": float(corr.mean() / std),
            "max_drawdown": float((np.maximum.accumulate(cumulative) - cumulative).max()),
        }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    log("Done: three validated artifacts prepared; nothing uploaded or staked.")


if __name__ == "__main__":
    main()
