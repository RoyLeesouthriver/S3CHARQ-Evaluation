"""Per-image metrics and empirical (linear interpolated) percentiles."""
import numpy as np
import torch


def image_metrics(original, reconstruction, lpips_net, normalize=False):
    mse = (original.float() - reconstruction.float()).square().flatten(1).mean(1)
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    quality = lpips_net(original.float(), reconstruction.float(), normalize=normalize).flatten()
    if not torch.isfinite(psnr).all() or not torch.isfinite(quality).all():
        raise ValueError('Nonfinite image metrics')
    return psnr.cpu().numpy(), quality.cpu().numpy()


def summarize(rows, target):
    psnr = np.array([r['psnr'] for r in rows], dtype=np.float64)
    lpips = np.array([r['lpips'] for r in rows], dtype=np.float64)
    return dict(count=len(rows), psnr_mean=float(psnr.mean()),
                psnr_p03=float(np.percentile(psnr, 3)), lpips_mean=float(lpips.mean()),
                lpips_p97=float(np.percentile(lpips, 97)),
                outage_ratio=float(np.mean(lpips > target)))

