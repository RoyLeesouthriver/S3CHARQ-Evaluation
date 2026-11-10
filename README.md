# S3CHARQ Evaluation

> **Download status:** Source code and summary results are published here. The
> pretrained checkpoint and per-image CSV files are not yet uploaded to GitHub.
> Evaluation requires `checkpoint/s3charq_cifar10.pth`; cloning this repository
> alone is insufficient. See [release asset instructions](RELEASE_ASSETS.md).


Standalone **CIFAR10 / 32×32 / AWGN / one retransmission** evaluation extracted
from the existing implementation of S3CHARQ.

## Quick start

Original training environment (provided by the authors): NVIDIA RTX4090 in
Ubuntu 20.04 LTS with Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz,
under a PyTorch 1.12.1 environment (corrected by the authors).

Separately, the locally tested evaluation environment was Python 3.11.0,
PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121,
NVIDIA RTX 2070 SUPER (8 GB). Those reference-run versions are retained in
`requirements-modern.txt`. The default `requirements.txt` now targets Python 3.8
with the authors' corrected PyTorch 1.12.1 environment. The torchvision and other
dependency pins are compatibility selections, not claims about the exact original
training environment. The original CUDA build has not been specified.

```bash
conda create -n s3charq-eval python=3.8
conda activate s3charq-eval
python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt

# From this directory: download CIFAR10 and evaluate the first 1000 test images.
python evaluate.py --download --max-samples 1000 --output-dir my_smoke_run

# Full CIFAR10 test set (10000 images per SNR):
python evaluate.py --download --max-samples 0 --output-dir my_full_run

# Reuse an already downloaded torchvision CIFAR10 dataset:
python evaluate.py --data-root /path/to/dataset --max-samples 1000 --output-dir my_run
```

The torch/torchvision pairing follows the [official PyTorch previous-version
installation instructions](https://pytorch.org/get-started/previous-versions/).
Do not reinstall torch blindly in an existing working training environment.
First inspect the interpreter actually executing the script:

```bash
python -c "import sys, torch, torchvision, inspect; print(sys.executable); print(sys.version); print(torch.__version__, torchvision.__version__, torch.version.cuda); print(inspect.signature(torch.load))"
```

### Older PyTorch checkpoint loading

`TypeError: 'weights_only' is an invalid keyword argument for Unpickler()` means
the active `torch.load` interface does not expose `weights_only`; it forwards
that keyword to Python's unpickler. This is not itself a tensor shape or checkpoint
corruption error. The active environment may differ from the recorded training
environment; the evaluator now prints actual Python/torch/torchvision/CUDA versions.

`compat.py` detects the explicit argument before using it. New torch versions
retain restricted loading (`weights_only=True`). Old versions omit the argument,
but automatically accept only the exact distributed checkpoint SHA256. For a
different checkpoint, old pickle loading requires the explicit
`--trust-legacy-checkpoint` flag: use it only for a file you trust. A failed
restricted load is never silently retried with an unrestricted loader. All model
state dictionaries still load strictly. The distributed checkpoint is unchanged;
you do not need to retrain or convert it for this interface fix.

The legacy dependency pins avoid the Python-3.11-only NumPy/Pillow versions used
in the previous local validation. Backend TF32 flags are also guarded for older
torch interfaces. Python 3.8 syntax and simulated old/new loading branches are
checked locally; a full Ubuntu/Python 3.8/PyTorch 1.12.1 run has **not** been performed
on the authors' remote server. Reference results still describe the original
local environment, not a new measurement under PyTorch 1.12.1.

For an existing environment, first replace the code and try a small run without
changing packages:

```bash
python evaluate.py --data-root /path/to/dataset --max-samples 64 --snrs 1 --output-dir compatibility_smoke
```

`--download` is optional and explicitly enables CIFAR10 download. LPIPS also
requires pretrained VGG16 weights: the first initialization may download about
528 MiB into the standard torch hub cache. This repository does **not** include
CIFAR10 or VGG16. An offline reviewer must prepopulate both caches. The small
learned LPIPS calibration weights are supplied by the installed `lpips` package.

Use `--device cpu` for CPU inference (much slower), or `--device cuda:0` for GPU.
Lower `--batch-size` on smaller GPUs. A one-image, batch-size-one CPU smoke test
also passed; the full reference run was performed on GPU. Reference bitwise equivalence
was checked on the local GPU, not across all platforms.

`config.yaml` is written in the JSON-compatible subset of YAML 1.2. Edit it with
JSON syntax (double quotes; no YAML comments); the evaluator reads it with the
standard library, so PyYAML is not needed. Relative paths resolve against the
configuration file directory, not the shell's current directory. CLI flags override
the corresponding fields. An existing `metrics.csv` is never silently overwritten.

## What is included

```text
checkpoint/s3charq_cifar10.pth   versioned bundle: HARQ, PPO, standalone SwinJSCC
checkpoint/provenance.json     source filenames, SHA256, and provenance limitation
models/s3charq.py              JS3C_HARQ_LPIPS integration
models/swinjscc.py             standalone SwinJSCCcomplex baseline
models/check_encoder.py        original auxiliary-feature compression network
models/lpips_estimator.py      original LPIPS estimator
models/retransmission.py       entropy/compression/fusion networks
models/ppo_agent.py            inference ActorCritic (not the optimizer)
models/blocks.py               required original helper layers
models/*swinvit.py             required original Swin backbone definitions
channel.py                    original normalization and channel operations
dataset.py                    CIFAR10 test loader
metrics.py                    per-image metrics and empirical percentiles
evaluate.py                   evaluation CLI
reference_results.csv         actual full-test-set results (10000 images per SNR)
full_reference_run/           full-run raw observations, metadata, aggregate results
reference_run/                additional 1000-image-per-SNR smoke test
extraction_validation.json    original-versus-extracted forward comparison
tests/                        lightweight metric and observation tests
```

The extra backbone/helper files are necessary dependencies, not copies of the
entire training project. The single `.pth` file contains three state dictionaries
(`harq`, `ppo`, `swin`), not only JSCC weights. All three are loaded with
`strict=True`; no missing tensors, shape mismatches, or random initializations
are silently accepted. Use only trusted checkpoints.

## Evaluation protocol

- CIFAR10 **test** split; fixed dataset order; RGB [0,1]; no crop, augmentation,
  input normalization, or reconstruction clipping.
- AWGN SNRs: 1, 4, 7, 10, 13 dB. Native model masking ratio: 0.125 for each
  transmission. This parameter is not an independently audited physical total
  bandwidth ratio including auxiliary symbols.
- FP32; AMP and TF32 disabled. Models and LPIPS are in evaluation mode.
- Outage: **final true LPIPS > 0.064**. Default rule and policy-state thresholds
  are also 0.064, but remain separate config fields.
- LPIPS: VGG, `normalize=False`, preserving the source evaluation convention.
  This means the supplied [0,1] images are **not** remapped to [-1,1]. Changing
  `lpips_normalize` changes the metric; the pretrained predictor, threshold, and
  policy must then be recalibrated. Do not compare differently normalized runs.
- PSNR: per-image RGB MSE with `data_range=1`; average the per-image dB values.
  Zero MSE is capped at 120 dB for numerical stability.
- Long tail: 3rd percentile PSNR, 97th percentile LPIPS, using NumPy's default
  linear interpolation. Pooled percentiles are recomputed from all samples,
  not averaged from per-SNR percentiles.

Methods:

| Output name | Decision |
|---|---|
| S3CHARQ | PPO actor; action 1 selects retransmission |
| S3CHARQ- | Retransmit if predicted first-round LPIPS exceeds rule threshold |
| SwinJSCC | Independently trained standalone first-transmission baseline |
| No-retransmission | HARQ model's first reconstruction, not standalone SwinJSCC |
| Always-retransmit | HARQ model's second reconstruction for all images |

PPO, static threshold and the two HARQ diagnostics share exactly the same first/
second reconstructions. Their difference is solely the selection decision.
Both reconstructions are precomputed for measurement, even for a stop action;
wall-clock runtime is therefore **not** a deployment latency measurement.
Different retransmission probabilities imply different average resource use.

PPO observation ordering is unchanged:
`[SNR/13, ratio, state_threshold, predicted_LPIPS, predicted_LPIPS-state_threshold, xcom(64)]`.
Only predicted quality and model features are used for actions; true quality is
used only for reporting. The default is stochastic policy sampling, matching
the current original `PPOAgent.select_action` implementation (which samples even
with `training=False`). Use `--policy-action argmax` for deterministic actions,
but label those results separately.


## Outputs and tests

`metrics.csv`: one row per method/SNR, count, mean PSNR, PSNR P03, mean LPIPS,
LPIPS P97 and outage ratio (in [0,1]). Retransmission rates are not reported.
`overall_metrics.csv`: pooled equal-SNR sample statistics.
`raw_samples.csv`: individual test index, PSNR, LPIPS, decision and outage.
`run_metadata.json`: effective configuration, environment and checkpoint SHA256.

Actual pooled reference results (10000 images × five SNRs, one seed):

| Method | Mean PSNR | PSNR P03 | Mean LPIPS | LPIPS P97 | Outage |
|---|---:|---:|---:|---:|---:|
| SwinJSCC | 31.02 | 26.08 | 0.047 | 0.093 | 15.64% |
| S3CHARQ | 32.08 | 27.17 | 0.034 | 0.068 | 4.31% |
| S3CHARQ- | 32.02 | 27.07 | 0.035 | 0.070 | 4.89% |


```bash
python -m unittest discover -s tests -v
```

