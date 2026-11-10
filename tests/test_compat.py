import unittest
from unittest.mock import patch
import compat


class CompatibilityTests(unittest.TestCase):
    def test_modern_uses_restricted_loader(self):
        def modern(path, map_location=None, *, weights_only=False):
            self.assertTrue(weights_only)
            self.assertEqual(map_location, 'cpu')
            return {'format_version': 1}
        with patch.object(compat.torch, 'load', modern):
            self.assertEqual(compat.load_checkpoint('unused')['format_version'], 1)

    def test_old_torch_does_not_receive_unknown_keyword(self):
        def legacy(path, map_location=None, **pickle_load_args):
            self.assertEqual(pickle_load_args, {})
            return {'format_version': 1}
        with patch.object(compat.torch, 'load', legacy), patch.object(
                compat, 'checkpoint_sha256', return_value=compat.BUNDLED_SHA256):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(compat.load_checkpoint('unused')['format_version'], 1)

    def test_old_torch_rejects_unknown_checkpoint_by_default(self):
        def legacy(path, map_location=None, **pickle_load_args):
            self.fail('Unsafe loading should not be reached')
        with patch.object(compat.torch, 'load', legacy), patch.object(
                compat, 'checkpoint_sha256', return_value='different'):
            with self.assertRaises(RuntimeError):
                compat.load_checkpoint('unused')

    def test_restricted_load_failure_does_not_fall_back(self):
        def modern(path, map_location=None, *, weights_only=False):
            raise ValueError('Rejected checkpoint')
        with patch.object(compat.torch, 'load', modern):
            with self.assertRaises(ValueError):
                compat.load_checkpoint('unused', True)

    def test_explicit_trust_for_old_custom_checkpoint(self):
        def legacy(path, map_location=None, **pickle_load_args):
            return {'trusted': True}
        with patch.object(compat.torch, 'load', legacy):
            with self.assertWarns(RuntimeWarning):
                self.assertTrue(compat.load_checkpoint('unused', True)['trusted'])


if __name__ == '__main__':
    unittest.main()

