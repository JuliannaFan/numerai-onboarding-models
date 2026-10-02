# Third-party notices

This project adapts the official [numerai/example-scripts](https://github.com/numerai/example-scripts) tutorials, licensed under the MIT License, copyright (c) 2022 Numerai. The original license is included in `LICENSE`.

Upstream sources, pinned to commit `856c7e7352057398b04847710a38c6d968db405b` inspected on October 1, 2026:

- [Hello Numerai](https://github.com/numerai/example-scripts/blob/856c7e7352057398b04847710a38c6d968db405b/numerai/hello_numerai.ipynb)
- [Feature Neutralization](https://github.com/numerai/example-scripts/blob/856c7e7352057398b04847710a38c6d968db405b/numerai/feature_neutralization.ipynb)
- [Target Ensemble](https://github.com/numerai/example-scripts/blob/856c7e7352057398b04847710a38c6d968db405b/numerai/target_ensemble.ipynb)

Adaptations include bounded Parquet downloading, removal of constant training features, explicit random seeds and CPU limits, least-squares feature neutralization, pinned Python 3.12 dependencies, serialized-model checks, and evaluation across all eligible validation eras.

The trained LightGBM parameters were produced locally by running these adaptations. Training datasets are not included. Dependencies are installed separately through the requirements files and retain their respective licenses. Numerai's dataset rights and service terms are separate from this repository's MIT code license.
