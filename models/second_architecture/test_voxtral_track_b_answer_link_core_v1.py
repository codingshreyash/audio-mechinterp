"""Fake-model tests only: no Voxtral checkpoint, corpus audio, or candidate inference."""

from __future__ import annotations

import types
import unittest
import math

import torch

from models.second_architecture import voxtral_track_b_answer_link_core_v1 as core


class FakeInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_modal_projector = torch.nn.Identity()


class FakeVoxtral(torch.nn.Module):
    """Projector is called once, flat [B*T,D]; LM logits depend on its output."""

    def __init__(self):
        super().__init__()
        self.model = FakeInner()
        self.last_keep = None
        self.last_mask = None

    def forward(self, *, input_ids, attention_mask, input_features,
                use_cache=False, logits_to_keep=1):
        b, t, width = input_features.shape
        flat = self.model.multi_modal_projector(input_features.reshape(b * t, width))
        x = flat.reshape(b, t, width)[:, :, 0].sum(1)
        length = input_ids.shape[1]
        position = torch.arange(length, device=x.device, dtype=torch.float32)[None, :]
        logits = torch.zeros((b, length, 32), device=x.device)
        logits[:, :, 7] = x[:, None] * 0.9 + position * 0.02
        logits[:, :, 8] = -x[:, None] * 0.7 + position * 0.01
        logits[:, :, 9] = input_ids.float() * 0.03
        logits[:, :, core.EOS] = x[:, None] * 0.2 - position * 0.015
        self.last_keep = logits_to_keep
        self.last_mask = attention_mask.detach().clone()
        return types.SimpleNamespace(logits=logits[:, -logits_to_keep:, :])

    def generate(self, *, input_ids, attention_mask, input_features,
                 do_sample, use_cache, max_new_tokens, eos_token_id, pad_token_id):
        assert not do_sample and use_cache and max_new_tokens == 32 and eos_token_id == 2
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                              input_features=input_features, use_cache=True,
                              logits_to_keep=1).logits[0, -1]
        token = 7 if logits[7] > logits[8] else 8
        tail = torch.tensor([[token, eos_token_id]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, tail], dim=1)


class FakeProcessor:
    tokenizer = types.SimpleNamespace(all_special_ids=[2, 24])

    def batch_decode(self, sequences, *, skip_special_tokens):
        names = {7: 'Banana', 8: 'Tomato', 9: 'Extra'}
        out = []
        for seq in sequences:
            tokens = [names.get(t, '' if skip_special_tokens else f'<{t}>') for t in seq]
            out.append(''.join(tokens))
        return out


def fixture():
    model = FakeVoxtral()
    source = torch.zeros((1, 3, core.PROJECTOR_WIDTH), dtype=torch.float32)
    source[0, 1, 0] = 1.0
    source[0, 1, 1] = -0.0
    donor = source.clone()
    donor[0, 1, 0] = -2.0
    batch = {'input_ids': torch.tensor([[24, 24, 24, 5]]),
             'attention_mask': torch.ones((1, 4), dtype=torch.long),
             'input_features': source}
    return model, batch, source[0], donor[0]


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(ch) for ch in text]


class CaseInsensitiveTokenizer(TinyTokenizer):
    def encode(self, text, add_special_tokens=False):
        return super().encode(text.lower(), add_special_tokens=add_special_tokens)


class CoreTests(unittest.TestCase):
    def test_contextual_alias_freeze_rejects_boundary_change(self):
        tok = TinyTokenizer()
        prompt = [24, 24, 5]
        aliases = core.freeze_aliases(tok, 'banana', prompt_ids=prompt,
                                      contextual_encode=lambda s: prompt + tok.encode(s))
        self.assertEqual(len(aliases), 6)
        self.assertTrue(all(a[-1] == core.EOS for a in aliases))
        expected = ('banana', ' banana', 'Banana', ' Banana', 'BANANA', ' BANANA')
        inventory = core.freeze_alias_inventory(tok, 'banana', prompt_ids=prompt,
                                                 contextual_encode=lambda s: prompt + tok.encode(s))
        self.assertEqual(tuple(row['surface'] for row in inventory), expected)
        self.assertEqual(aliases, tuple(row['token_ids_with_eos'] for row in inventory))
        self.assertEqual([a[0] for a in aliases], [98, 32, 66, 32, 66, 32])
        insensitive = CaseInsensitiveTokenizer()
        collapsed = core.freeze_alias_inventory(insensitive, 'banana', prompt_ids=prompt,
                                                 contextual_encode=lambda s: prompt + insensitive.encode(s))
        self.assertEqual([r['surface'] for r in collapsed], ['banana', ' banana'])
        with self.assertRaisesRegex(ValueError, 'boundary'):
            core.freeze_aliases(tok, 'banana', prompt_ids=prompt,
                                contextual_encode=lambda s: prompt[:-1] + tok.encode(s))

    def test_rows_and_direction_are_fixed_and_nonempty(self):
        self.assertEqual(core.word_rows(400, 1900, 2400, 3, 2), (0, 1))
        with self.assertRaisesRegex(ValueError, 'Empty'):
            core.word_rows(1000, 1100, 2400, 3, 2)
        with self.assertRaisesRegex(ValueError, 'padded'):
            core.word_rows(400, 1900, 2400, 3, 1)
        with self.assertRaisesRegex(ValueError, 'out-of-range'):
            core.checked_rows((1, 3), 3)
        _model, _batch, source, donor = fixture()
        d = core.mean_direction(source, donor, (1,))
        self.assertEqual(float(d[0]), -3.0)
        self.assertEqual(float(torch.linalg.vector_norm(d[1:])), 0.0)
        u1, info1 = core.random_control(d, 'pair_00|piper|banana')
        u2, info2 = core.random_control(d, 'pair_00|piper|banana')
        self.assertTrue(core.exact_equal(u1, u2))
        self.assertEqual(info1, info2)
        self.assertAlmostEqual(float(torch.dot(u1, d)), 0.0, places=5)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(u1)), float(torch.linalg.vector_norm(d)), places=5)

    def test_flat_hook_edits_every_alias_and_preserves_outside(self):
        model, _batch, source, donor = fixture()
        d = core.mean_direction(source, donor, (1,))
        with core.ProjectorHook(model, source, 2, 3, core.Edit('vector', (1,), d, 0.1)) as hook:
            result = model.model.multi_modal_projector(source.repeat(2, 1))
            self.assertEqual(result.shape, (6, core.PROJECTOR_WIDTH))
            self.assertTrue(core.exact_equal(result[0], source[0]))
            self.assertTrue(core.exact_equal(result[2], source[2]))
            self.assertTrue(core.exact_equal(result[3], source[0]))
            self.assertTrue(core.exact_equal(result[5], source[2]))
            self.assertTrue(core.exact_equal(result[1], result[4]))
            self.assertAlmostEqual(float(result[1, 0]), 0.7, places=6)
        self.assertEqual(hook.evidence()['flat_shape'], [6, core.PROJECTOR_WIDTH])
        changed = source.repeat(2, 1)
        changed[4, 0] += 0.01
        with self.assertRaisesRegex(ValueError, 'Alias 1 prepatch'):
            with core.ProjectorHook(model, source, 2, 3, core.Edit('zero')):
                model.model.multi_modal_projector(changed)
        with self.assertRaisesRegex(ValueError, 'Multiple projector calls'):
            with core.ProjectorHook(model, source, 1, 3, core.Edit('zero')):
                model.model.multi_modal_projector(source)
                model.model.multi_modal_projector(source)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            with core.ProjectorHook(model, source, 1, 3, core.Edit('zero')):
                pass

    def test_fixed_projector_width_rejected_in_hook_and_capture(self):
        model, batch, source, donor = fixture()
        narrow = source[:, :4].clone()
        with self.assertRaisesRegex(ValueError, 'Source must'):
            core.ProjectorHook(model, narrow, 1, 3, core.Edit('zero'))
        with self.assertRaisesRegex(ValueError, 'finite FP32'):
            core.mean_direction(narrow, donor[:, :4], (1,))
        narrowed_batch = dict(batch, input_features=batch['input_features'][:, :, :4].clone())
        with self.assertRaisesRegex(ValueError, 'Passive projector geometry'):
            core.capture_projector(model, narrowed_batch)

    def test_zero_and_full_donor_complete_generation(self):
        model, batch, source, donor = fixture()
        processor = FakeProcessor()
        common = dict(recipient='banana', donor_word='tomato', pad=0,
                      recipient_aliases=((7, 2),), donor_aliases=((8, 2),))
        clean = core.generate_answer(model, processor, batch, source=None, edit=None, **common)
        zero = core.generate_answer(model, processor, batch, source=source,
                                    edit=core.Edit('zero'), **common)
        full = core.generate_answer(model, processor, batch, source=source,
                                    edit=core.Edit('full', donor=donor), **common)
        self.assertEqual(clean['suffix_ids'], zero['suffix_ids'])
        self.assertEqual(clean['class'], 'recipient')
        self.assertTrue(zero['recipient_alias_covered'])
        self.assertEqual(zero['hook']['pre_hashes'], zero['hook']['post_hashes'])
        self.assertEqual(full['class'], 'opposite')
        self.assertTrue(full['donor_alias_covered'])
        self.assertEqual(full['hook']['hook_calls'], 1)
        donor_batch = dict(batch, input_features=donor[None])
        natural_donor = core.generate_answer(model, processor, donor_batch,
                                             recipient='tomato', donor_word='banana',
                                             source=None, edit=None, pad=0)
        core.assert_generation_controls(clean, zero, natural_donor, full)
        with self.assertRaisesRegex(ValueError, 'Full-projector donor'):
            core.assert_generation_controls(clean, zero, clean, full)

    def test_variable_alias_lengths_eos_and_sequential_class_mass(self):
        model, batch, source, _donor = fixture()
        aliases = ((7, 2), (9, 7, 2))
        scored = core.score_alias_class(model, batch, aliases, source, core.Edit('zero'), pad=0)
        self.assertEqual(scored['forward_count'], 2)
        self.assertEqual(scored['alias_lengths_including_eos'], [2, 3])
        self.assertEqual(scored['full_input_shapes'], [[1, 5], [1, 6]])
        self.assertEqual(scored['answer_logits_shapes'], [[1, 2, 32], [1, 3, 32]])
        self.assertEqual(model.last_keep, 3)
        self.assertEqual(model.last_mask[0].tolist(), [1, 1, 1, 1, 1, 1])
        self.assertEqual([len(x) for x in scored['token_logprobs_including_eos']], [2, 3])
        self.assertEqual([h['hook_calls'] for h in scored['hooks']], [1, 1])
        for idx, alias in enumerate(aliases):
            no_hook = core.score_alias_class(model, batch, (alias,), source, None, pad=0)
            self.assertAlmostEqual(scored['alias_logprobs'][idx], no_hook['alias_logprobs'][0], places=5)
            self.assertAlmostEqual(scored['token_logprobs_including_eos'][idx][-1],
                                   no_hook['token_logprobs_including_eos'][0][-1], places=5)
        expected = torch.logsumexp(torch.tensor(scored['alias_logprobs'], dtype=torch.float64), 0)
        self.assertAlmostEqual(scored['log_mass'], float(expected), places=7)
        grad = core.score_alias_class(model, batch, aliases, source, core.Edit('zero'),
                                      pad=0, grad=True, gradient_rows=(1,))
        core.assert_score_parity(grad, scored)
        self.assertEqual(grad['gradient_accumulation_dtype'], 'torch.float32')
        self.assertEqual(grad['alias_weight_dtype'], 'torch.float64')
        self.assertEqual(grad['gradient_rows'], [1])
        self.assertEqual(len(grad['per_alias_selected_gradients']), 2)
        self.assertTrue(all(g.shape == (1, core.PROJECTOR_WIDTH)
                            for g in grad['per_alias_selected_gradients']))
        self.assertTrue(all(grad['per_alias_gradient_finite']))
        self.assertTrue(all(grad['per_alias_full_gradient_finite']))
        self.assertTrue(all(math.isfinite(n) and n >= 0 for n in grad['per_alias_full_gradient_l2']))
        restored = sum((w * g for w, g in zip(grad['alias_softmax_weights'],
                                             grad['per_alias_selected_gradients'])),
                       torch.zeros_like(grad['per_alias_selected_gradients'][0]))
        self.assertTrue(torch.allclose(restored, grad['gradient'][[1]].cpu(), atol=1e-6))
        with self.assertRaisesRegex(ValueError, 'likelihood'):
            tampered = dict(scored, token_logprobs_including_eos=[[-99.0, -99.0],
                                                                  scored['token_logprobs_including_eos'][1]])
            core.assert_score_parity(grad, tampered)
        with self.assertRaisesRegex(ValueError, 'Invalid frozen'):
            core.score_alias_class(model, batch, (aliases[0], aliases[0]), source,
                                   core.Edit('zero'), pad=0)

    def test_answer_gradient_predicts_local_full_reply_margin(self):
        model, batch, source, donor = fixture()
        direction = core.mean_direction(source, donor, (1,))
        r_aliases = ((7, 2), (9, 7, 2))
        d_aliases = ((8, 2), (9, 8, 2))
        result = core.answer_jacobian(model, batch, r_aliases, d_aliases,
                                      source, direction, (1,), pad=0)
        self.assertTrue(math.isfinite(result['j']))
        self.assertEqual(result['recipient']['forward_count'], 2)
        self.assertEqual(result['donor']['forward_count'], 2)
        plus_r = core.score_alias_class(model, batch, r_aliases, source,
                                        core.Edit('vector', (1,), direction, 0.001), pad=0)
        plus_d = core.score_alias_class(model, batch, d_aliases, source,
                                        core.Edit('vector', (1,), direction, 0.001), pad=0)
        measured = plus_d['log_mass'] - plus_r['log_mass'] - result['s0']
        self.assertAlmostEqual(measured, 0.001 * result['j'], delta=2e-4)

    def test_predicted_switch_requires_every_crossing_clause(self):
        model, batch, source, donor = fixture()
        proc = FakeProcessor()
        common = dict(recipient='banana', donor_word='tomato', pad=0,
                      recipient_aliases=((7, 2),), donor_aliases=((8, 2),))
        clean = core.generate_answer(model, proc, batch, source=None, edit=None, **common)
        zero = core.generate_answer(model, proc, batch, source=source,
                                    edit=core.Edit('zero'), **common)
        edited = core.generate_answer(model, proc, batch, source=source,
                                      edit=core.Edit('full', donor=donor), **common)
        good = core.predicted_switch_clauses(clean, zero, edited,
                                             s0=-0.2, j=3.0, alpha=0.1, measured_s=0.1)
        self.assertTrue(good['predicted_complete_answer_switch'])
        no_cross = core.predicted_switch_clauses(clean, zero, edited,
                                                 s0=-0.4, j=3.0, alpha=0.1, measured_s=0.1)
        self.assertFalse(no_cross['predicted_margin_positive'])
        self.assertFalse(no_cross['predicted_complete_answer_switch'])
        uncovered = dict(edited, donor_alias_covered=False)
        self.assertFalse(core.predicted_switch_clauses(clean, zero, uncovered,
                         s0=-0.2, j=3.0, alpha=0.1, measured_s=0.1)
                         ['predicted_complete_answer_switch'])

    def test_strict_complete_reply_classes(self):
        ids = [7, 2]
        self.assertEqual(core.classify(ids, 'Banana', {2, 24}, 2, 'banana', 'tomato'), 'recipient')
        self.assertEqual(core.classify([8, 2], 'Tomato', {2, 24}, 2, 'banana', 'tomato'), 'opposite')
        self.assertEqual(core.classify([7], 'Banana', {2, 24}, 2, 'banana', 'tomato'), 'missing_eos')
        self.assertEqual(core.classify([24, 2], 'Banana', {2, 24}, 2, 'banana', 'tomato'), 'malformed_special')
        self.assertEqual(core.classify([9, 2], 'Two words', {2, 24}, 2, 'banana', 'tomato'), 'malformed_surface')


if __name__ == '__main__':
    unittest.main()
