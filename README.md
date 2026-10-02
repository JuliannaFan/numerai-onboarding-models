# Numerai Onboarding Models

Three working LightGBM prediction models adapted from Numerai's official tutorials: a baseline, a feature-neutralized model, and a two-target ensemble. The repository includes training code, downloadable Python 3.12 model files, and historical validation results.

All three original model uploads generated successful live submissions on Numerai on October 1, 2026. These are educational tutorial adaptations, not a claim of a novel strategy or proven live profitability.

## Models

| File | Method |
| --- | --- |
| [hello_numerai.pkl](models/hello_numerai.pkl) | LightGBM trained on `target_ender_60`; percentile-ranked predictions |
| [feature_neutralization.pkl](models/feature_neutralization.pkl) | Baseline predictions neutralized against the medium/agility feature intersection |
| [target_ensemble.pkl](models/target_ensemble.pkl) | Average percentile ranks of models trained on `target_ender_60` and `target_teager2b_60` |

Each callable accepts `live_features` and an optional `live_benchmark_models` argument. It returns a DataFrame with the original stock IDs and one finite, varying `prediction` column in `[0, 1]`. The neutralization and ranking operate on the supplied batch; pass one era at a time when evaluating historical data.

## Historical validation

Training used Numerai v5.3, 2,746,268 rows across 574 eras, and 267 nonconstant features from the Faith feature set. Each tree model has 2,000 estimators, learning rate 0.01, maximum depth 5, 31 leaves, and feature sampling of 0.1. Validation excludes test rows, missing targets, and the first 12 eras after the training period.

| Model | Validation eras | Mean CORR | CORR standard deviation | Mean / standard deviation | Maximum cumulative-CORR drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 640 | 0.009827 | 0.012140 | 0.809452 | 0.103057 |
| Feature neutralization | 640 | 0.010112 | 0.012341 | 0.819410 | 0.115581 |
| Target ensemble | 640 | 0.010121 | 0.012339 | 0.820207 | 0.089316 |

These are local historical validation metrics, not annualized investment Sharpe ratios, live tournament scores, or returns. See [the exact summary](reports/validation_summary.json) and [per-era results](reports/validation_per_era.csv). Hosted Numerai diagnostics are separate from these local results.

## Reproduce training and validation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv venv --python 3.12 .venv312
uv pip install --python .venv312/bin/python -r requirements.lock.txt
sh run_models.sh
```

The script downloads data through Numerai's API into the ignored `data/` directory. Remote Parquet column projection and bounded batches reduce transfer and memory usage. Training uses four CPU threads and may take time. Generated models, predictions, metadata, and reports are written to the ignored `artifacts/` directory. Existing trained text models in that directory are reused.

The published models are in `models/`; their SHA-256 hashes and runtime versions are recorded in [models/manifest.json](models/manifest.json). To rebuild this publication set after training, run `export_models.py` in the same environment. It removes absolute source paths from pickle metadata, verifies identical predictions, and exports only the models and derived reports.

## Load a model

Use Python 3.12 with the pinned dependencies. Only unpickle files from a source you trust: pickle deserialization can execute code.

```python
from pathlib import Path
import cloudpickle
import pandas as pd

predict = cloudpickle.loads(Path("models/hello_numerai.pkl").read_bytes())
live_features = pd.read_parquet("path/to/current-live.parquet")
predictions = predict(live_features, None)
predictions.to_csv("submission.csv")
```

On macOS, the launcher uses the OpenMP library bundled with the scikit-learn wheel. For a separate Python command, use the same library path:

```sh
export DYLD_LIBRARY_PATH="$PWD/.venv312/lib/python3.12/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

The onboarding server reported `numerai-predict:f1a3f48 - Python 3.12`. Dependencies match [that deployed inference environment](https://github.com/numerai/numerai-predict/blob/f1a3f48/py3.12/requirements.txt). Python 3.13 pickles failed in that environment; repackaging the trained trees with Python 3.12 resolved execution. Check the currently selected runtime when uploading elsewhere.

## Data and licensing

This repository contains model parameters, implementation code, and derived evaluation scores. Numerai datasets, row-level predictions, account records, API keys, and authorization files are excluded. Obtain data directly from Numerai under its [Terms of Service](https://numer.ai/terms); the repository's code license does not grant rights to Numerai's data.

The code is distributed under the [MIT License](LICENSE), retaining Numerai's original copyright notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the upstream tutorials and the adaptations made here. This project is not endorsed by Numerai.
