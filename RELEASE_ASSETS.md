# Release assets (upload pending)

The source repository intentionally excludes the approximately 1 GB pretrained
checkpoint and the full per-image CSVs. No download is claimed to be available yet.
The maintainer has a complete, locally verified distribution archive named
`S3CHARQ-Evaluation-Release.zip` (949499323 bytes).

Archive SHA256:
`d7dca2c73b91c5778a1967839c03d7ddc7238f03dbe06debe0f64e2f06ec4bee`

Checkpoint SHA256:
`528b757482c5707ed6d4a253e4abd1814a6f5d3a4ff226ae50efe7098044c3e3`

Maintainer steps:
1. Create a GitHub Release and upload the full archive as a release asset.
2. Add the resulting asset link to this page and remove the pending notice.
3. Reviewers can extract the archive and run it independently, or copy its
   `checkpoint/s3charq_cifar10.pth` into a clone of this repository.

The archive includes the original supplied README, unchanged model weights,
0.064 threshold reference results and per-image measurements. The GitHub README
adds this distribution-status notice. The complete archive also carries its own
SHA256SUMS.json; that manifest is separate from the source repository manifest.

