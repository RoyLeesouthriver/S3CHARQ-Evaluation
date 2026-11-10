"""Python 3.8 / older PyTorch compatibility without silent unsafe fallback."""
import hashlib
import inspect
import warnings
import torch

# Exact distributed tensor-only bundle, verified when the review package was built.
BUNDLED_SHA256 = '528b757482c5707ed6d4a253e4abd1814a6f5d3a4ff226ae50efe7098044c3e3'


def checkpoint_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path, trust_legacy_checkpoint=False):
    # Old torch.load accepts **pickle_load_args, which forwards unknown keywords
    # to pickle.Unpickler. Check explicit support before passing weights_only.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location='cpu', weights_only=True)
    if not trust_legacy_checkpoint and checkpoint_sha256(path) != BUNDLED_SHA256:
        raise RuntimeError(
            'This PyTorch lacks weights_only. The file does not match the supplied '
            'bundle SHA256. Use the original bundle, upgrade PyTorch, or explicitly '
            'pass --trust-legacy-checkpoint ONLY for a checkpoint you trust.')
    warnings.warn('Legacy torch.load: weights_only is unavailable. Loading a verified '
                  'bundled checkpoint or one explicitly trusted by the user. '
                  'Never load untrusted pickle files.', RuntimeWarning)
    return torch.load(path, map_location='cpu')


def configure_precision():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cuda_backend = getattr(torch.backends, 'cuda', None)
    matmul = getattr(cuda_backend, 'matmul', None)
    if matmul is not None and hasattr(matmul, 'allow_tf32'):
        matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False

