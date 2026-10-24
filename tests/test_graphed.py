"""Needs torch and transformers; the CUDA graph itself is checked only where there is a GPU."""
import os
import sys
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_transformer.graphed import GraphedGPT2


def tiny(device, dtype):
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=300, n_positions=64, n_embd=32, n_layer=3, n_head=4,
                        scale_attn_by_inverse_layer_idx=True, activation_function='gelu_new')
    return GPT2LMHeadModel(config).to(device, dtype).eval()


class GraphedTest(unittest.TestCase):
    def check(self, device, dtype, tolerance):
        model = tiny(device, dtype)
        fast = GraphedGPT2(model)
        seq = torch.randint(0, 300, (60,)).tolist()

        def expected(tokens):
            with torch.no_grad():
                return model(torch.tensor([tokens], device=device)).logits[0, -1].float()

        def feed(tokens, past=None):
            return fast(torch.tensor([tokens], device=device), past_key_values=past, use_cache=True)

        out = feed(seq[:20])   # a prompt in one pass
        torch.testing.assert_close(out.logits[0, -1], expected(seq[:20]), atol=tolerance, rtol=0)
        for n in range(20, 30):   # then a token at a time
            out = feed(seq[n:n + 1], out.past_key_values)
            torch.testing.assert_close(out.logits[0, -1], expected(seq[:n + 1]), atol=tolerance, rtol=0)
        out.past_key_values.crop(25)   # it forgets the last five and hears others instead
        out = feed(seq[40:43], out.past_key_values)
        torch.testing.assert_close(out.logits[0, -1], expected(seq[:25] + seq[40:43]), atol=tolerance, rtol=0)
        out = feed(seq[43:55], out.past_key_values)
        torch.testing.assert_close(out.logits[0, -1], expected(seq[:25] + seq[40:55]), atol=tolerance, rtol=0)
        self.assertEqual(out.past_key_values.length, 40)

    def test_it_gives_the_logits_of_the_model_it_wraps(self):
        self.check('cpu', torch.float32, 1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_its_cuda_graph_gives_them_too(self):
        self.check('cuda', torch.float16, 2e-2)


if __name__ == '__main__':
    unittest.main()
