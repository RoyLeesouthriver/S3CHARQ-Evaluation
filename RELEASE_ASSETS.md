# Release assets — v1.0.0

The complete evaluation package is now available:

- [Release page](https://github.com/RoyLeesouthriver/S3CHARQ-Evaluation/releases/tag/v1.0.0)
- [Download S3CHARQ-Evaluation-Release.zip](https://github.com/RoyLeesouthriver/S3CHARQ-Evaluation/releases/download/v1.0.0/S3CHARQ-Evaluation-Release.zip)
- Archive size: 949499323 bytes (approximately 949 MB).

The archive includes source code, pretrained HARQ/PPO/SwinJSCC weights,
0.064-threshold reference results and per-image measurements. CIFAR10 and the
pretrained VGG16 backbone used by LPIPS are not included.

## How to use

1. Download the ZIP asset linked above, not GitHub's automatically generated
   **Source code (zip)** or **Source code (tar.gz)**, which do not contain weights.
2. Extract the archive and enter its `S3CHARQ-Evaluation` directory.
3. Follow the README to install dependencies and evaluate.

If using a Git clone, copy `checkpoint/s3charq_cifar10.pth` from the extracted
package into the clone's `checkpoint/` directory. The full per-image CSVs are
available in `full_reference_run/` and `reference_run/` inside the archive.

## Integrity checks

Archive SHA256 (matches the digest reported by GitHub for the uploaded asset):

```text
d7dca2c73b91c5778a1967839c03d7ddc7238f03dbe06debe0f64e2f06ec4bee
```

Checkpoint SHA256:

```text
528b757482c5707ed6d4a253e4abd1814a6f5d3a4ff226ae50efe7098044c3e3
```

Linux:

```bash
sha256sum S3CHARQ-Evaluation-Release.zip
```

Windows PowerShell:

```powershell
Get-FileHash .\S3CHARQ-Evaluation-Release.zip -Algorithm SHA256
```

The archive retains its original README and `SHA256SUMS.json`. This documentation
update adds the published download links; it does not replace the ZIP asset,
modify model weights, change thresholds, or rerun evaluation.

