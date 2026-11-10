"""Standalone evaluation; no import from the original training project."""
import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import lpips
from tqdm import tqdm
from compat import load_checkpoint, configure_precision

from dataset import make_loader
from metrics import image_metrics, summarize
from models.s3charq import JS3C_HARQ_LPIPS
from models.swinjscc import SwinJSCCcomplex
from models.ppo_agent import ActorCritic

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_state(snr, ratio, threshold, prediction, xcom):
    pred = prediction.flatten()
    return torch.cat([torch.stack([torch.full_like(pred, snr / 13.0),
        torch.full_like(pred, ratio), torch.full_like(pred, threshold),
        pred, pred - threshold], 1), xcom], 1)


def parse_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    for key in ['checkpoint', 'data-root', 'output-dir', 'device']:
        parser.add_argument('--' + key)
    for key in ['batch-size', 'max-samples', 'num-workers', 'seed']:
        parser.add_argument('--' + key, type=int)
    parser.add_argument('--snrs', type=int, nargs='+')
    parser.add_argument('--policy-action', choices=['sample', 'argmax'])
    parser.add_argument('--download', action='store_true', default=None)
    parser.add_argument('--trust-legacy-checkpoint', action='store_true', default=None,
                        help='Allow old pickle loading for a custom checkpoint you trust')
    args = vars(parser.parse_args())
    # JSON is a YAML 1.2 subset. Deliberately avoids an extra PyYAML dependency.
    config_path = Path(args.pop('config')).resolve()
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.update({k: v for k, v in args.items() if v is not None})
    cfg.setdefault('trust_legacy_checkpoint', False)
    for key in ['checkpoint', 'data_root', 'output_dir']:
        path = Path(cfg[key]).expanduser()
        cfg[key] = str(path if path.is_absolute() else config_path.parent / path)
    if cfg['device'] == 'auto':
        cfg['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if cfg['batch_size'] <= 0 or cfg['max_samples'] < 0 or cfg['num_workers'] < 0:
        raise ValueError('Invalid batch/sample/worker count')
    if not cfg['snrs'] or len(set(cfg['snrs'])) != len(cfg['snrs']):
        raise ValueError('SNRs must be nonempty and unique')
    if any(int(s) != s or not 0 <= s <= 15 for s in cfg['snrs']):
        raise ValueError('The trained LPIPS estimator supports integer SNRs in [0,15]')
    if not 0.1 <= cfg['ratio'] <= 1 or cfg['policy_action'] not in ['sample', 'argmax']:
        raise ValueError('Invalid ratio or policy action')
    return cfg


def main():
    cfg = parse_config()
    seed_all(cfg['seed'])
    configure_precision()
    print('Runtime: Python {} | torch {} | torchvision {} | CUDA {}'.format(
        platform.python_version(), torch.__version__, torchvision.__version__,
        torch.version.cuda), flush=True)
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.csv').exists():
        raise FileExistsError('Choose a new output directory; metrics.csv already exists')
    bundle = load_checkpoint(cfg['checkpoint'], cfg['trust_legacy_checkpoint'])
    if bundle.get('format_version') != 1:
        raise ValueError('Expected the supplied version-1 evaluation bundle')
    models = {'harq': JS3C_HARQ_LPIPS(), 'swin': SwinJSCCcomplex(), 'ppo': ActorCritic(69)}
    for key, model in models.items():
        model.load_state_dict(bundle[key], strict=True)
        model.to(device).float().eval().requires_grad_(False)
    del bundle
    perceptual = lpips.LPIPS(net='vgg').to(device).eval().requires_grad_(False)
    loader = make_loader(cfg['data_root'], cfg['batch_size'], cfg['max_samples'],
                         cfg['num_workers'], cfg['download'])
    metadata = dict(config=cfg, actual_samples_per_snr=len(loader.dataset),
                    checkpoint_sha256=sha256(cfg['checkpoint']),
                    python=platform.python_version(), torch=torch.__version__,
                    torchvision=torchvision.__version__, numpy=np.__version__,
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                    channel='AWGN', precision='FP32; AMP/TF32 disabled',
                    note='Small subsets are smoke tests, not paper-reproduction evidence.')
    with open(out / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    methods = ['SwinJSCC', 'S3CHARQ', 'S3CHARQ-', 'Always-retransmit', 'No-retransmission']
    raw, summary = [], []
    with torch.no_grad():
        for snr in cfg['snrs']:
            # Independent, repeatable SNR runs. Dataset order is fixed.
            seed_all(cfg['seed'] + 1009 * int(snr))
            offset = 0
            for images, _ in tqdm(loader, desc='AWGN {:g} dB'.format(snr)):
                x = images.to(device).float()
                b = len(x)
                first, second, _, pred, _, _, xcom = models['harq'](
                    x, False, True, snr, cfg['ratio'], cfg['ratio'], 'AWGN')
                obs = build_state(snr, cfg['ratio'], cfg['state_threshold'], pred, xcom)
                logits, _ = models['ppo'](obs)
                action = (torch.distributions.Categorical(logits=logits).sample()
                          if cfg['policy_action'] == 'sample' else logits.argmax(1)).cpu().numpy()
                rule = (pred.flatten() > cfg['rule_threshold']).cpu().numpy()
                p1, l1 = image_metrics(x, first, perceptual, cfg['lpips_normalize'])
                p2, l2 = image_metrics(x, second, perceptual, cfg['lpips_normalize'])
                swin = models['swin'](x, False, True, snr, cfg['ratio'], 'AWGN')
                ps, ls = image_metrics(x, swin, perceptual, cfg['lpips_normalize'])
                choices = {'SwinJSCC': (ps, ls, np.zeros(b)),
                           'S3CHARQ': (np.where(action, p2, p1), np.where(action, l2, l1), action),
                           'S3CHARQ-': (np.where(rule, p2, p1), np.where(rule, l2, l1), rule),
                           'Always-retransmit': (p2, l2, np.ones(b)),
                           'No-retransmission': (p1, l1, np.zeros(b))}
                for name, (psnr, quality, acts) in choices.items():
                    for i in range(b):
                        raw.append(dict(method=name, snr=snr, image_index=offset+i,
                            psnr=float(psnr[i]), lpips=float(quality[i]), retransmit=int(acts[i]),
                            outage=int(quality[i] > cfg['target_lpips'])))
                offset += b
            for name in methods:
                row = dict(method=name, snr=snr, **summarize(
                    [r for r in raw if r['method'] == name and r['snr'] == snr], cfg['target_lpips']))
                summary.append(row)
                print('{method:20s} PSNR={psnr_mean:.3f} P03={psnr_p03:.3f} LPIPS={lpips_mean:.5f} '
                      'P97={lpips_p97:.5f} outage={outage_ratio:.2%}'.format(**row), flush=True)
            write_csv(out / 'metrics.csv', summary)
            write_csv(out / 'raw_samples.csv', raw)
    write_csv(out / 'overall_metrics.csv', [dict(method=name, **summarize(
        [r for r in raw if r['method'] == name], cfg['target_lpips'])) for name in methods])


if __name__ == '__main__':
    main()

