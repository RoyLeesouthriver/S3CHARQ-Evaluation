import unittest
import numpy as np
import torch
from metrics import summarize, image_metrics
from evaluate import build_state


class EvaluationTests(unittest.TestCase):
    def test_percentiles_and_outage_boundary(self):
        rows = [dict(psnr=i, lpips=i/100, retransmit=i % 2) for i in range(1, 101)]
        result = summarize(rows, .5)
        self.assertAlmostEqual(result['psnr_mean'], 50.5)
        self.assertAlmostEqual(result['psnr_p03'], 3.97)
        self.assertAlmostEqual(result['lpips_p97'], .9703)
        self.assertEqual(result['outage_ratio'], .5)
        self.assertNotIn('retransmission_ratio', result)

    def test_policy_state_layout(self):
        prediction = torch.tensor([[.1], [.04]])
        xcom = torch.zeros(2, 64)
        state = build_state(13, .125, .064, prediction, xcom)
        self.assertEqual(tuple(state.shape), (2, 69))
        np.testing.assert_allclose(state[0, :5].numpy(), [1, .125, .064, .1, .036], rtol=1e-6)

    def test_per_image_psnr(self):
        x = torch.zeros(2, 3, 4, 4)
        y = torch.stack([torch.full((3, 4, 4), .1), torch.full((3, 4, 4), .01)])
        def fake_lpips(a, b, normalize):
            return (a-b).abs().mean((1, 2, 3))
        psnr, quality = image_metrics(x, y, fake_lpips)
        np.testing.assert_allclose(psnr, [20, 40], atol=1e-5)
        np.testing.assert_allclose(quality, [.1, .01], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()

