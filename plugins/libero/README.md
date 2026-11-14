# LIBERO plugin for Miles π₀.₅

See [the π₀.₅ guide](../../docs/models/pi05/pi05.md) for installation, base model/data
downloads, training/resume, export, SGLang serving and LIBERO evaluation.

`pi05_tools` is an optional editable package. The core Miles model does not import
it. `setup.sh` installs pinned dependency checkouts and separate train/simulation
and serving environments into a chosen work directory. `env.sh` selects shared
asset caches. Keep downloads, weights and results under the ignored `local/`
directory or another data volume, not in this plugin's source tree.

- `convert_base`, `download_data`, `compute_norm_stats`, `prepare_libero`: official
  base conversion, pinned demonstrations, complete normalization statistics and manifests.
- `libero_data`, `libero_rollout`, `run_libero_sft`: data adapter and the Miles SFT launcher.
- `prepare_serving`, `libero`, `transforms`, `eval_libero`: matching inference transforms,
  SGLang action requests, simulator execution, resumable reports and optional videos.

OpenPI supplies tensor conversion helpers and data transforms; native Miles code
performs model training. No Transformers replacement files are installed.
