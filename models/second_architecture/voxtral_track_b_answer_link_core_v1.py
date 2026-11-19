"""Source-only model primitives for the reviewed Voxtral Track B V3+V4 pilot.

There is deliberately no CLI, checkpoint loader, corpus reader, or inference
entry point here. A separate reviewed admission/runner must supply a loaded
FP32 model, hash-bound batches, masks, QA receipt, and resource controls.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


PROMPT = ('What word did the speaker say between "said" and "and"? '
          'Answer with exactly one word. No explanation or punctuation.')
EOS = 2
MAX_NEW_TOKENS = 32
ALIAS_CASES = (str.lower, str.capitalize, str.upper)


def tensor_sha(value: torch.Tensor) -> str:
    """Hash dtype, shape, and raw bytes, distinguishing signed FP32 zero."""
    raw = value.detach().cpu().contiguous()
    return hashlib.sha256(
        f'{raw.dtype}|{tuple(raw.shape)}|'.encode() + raw.numpy().tobytes()
    ).hexdigest()


def exact_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and tensor_sha(a) == tensor_sha(b)


def word_rows(start: int, stop: int, frames: int, projector_rows: int) -> tuple[int, ...]:
    """Frozen nominal center rule; this is not a receptive-field claim."""
    if not (isinstance(start, int) and isinstance(stop, int) and
            isinstance(frames, int) and isinstance(projector_rows, int) and
            0 <= start < stop <= frames and projector_rows > 0):
        raise ValueError('Invalid PCM word slot or projector length')
    rows = tuple(r for r in range(projector_rows)
                 if start <= 160 * (8 * r + 3) < stop)
    if not rows or any(160 * (8 * r + 3) >= frames + 200 for r in rows):
        raise ValueError('Empty or padded word-row mask')
    return rows


def checked_rows(rows: Sequence[int], t: int) -> tuple[int, ...]:
    fixed = tuple(rows)
    if not fixed or fixed != tuple(sorted(set(fixed))) or any(
        not isinstance(r, int) or r < 0 or r >= t for r in fixed
    ):
        raise ValueError('Empty, unsorted, duplicate, or out-of-range rows')
    return fixed


def mean_direction(recipient: torch.Tensor, donor: torch.Tensor,
                   rows: Sequence[int]) -> torch.Tensor:
    if (recipient.ndim != 2 or recipient.shape != donor.shape or
            recipient.dtype != torch.float32 or donor.dtype != torch.float32 or
            not bool(torch.isfinite(recipient).all()) or
            not bool(torch.isfinite(donor).all())):
        raise ValueError('Natural projector states must be matched finite FP32 [T,D]')
    fixed = checked_rows(rows, recipient.shape[0])
    direction = donor[list(fixed)].mean(0) - recipient[list(fixed)].mean(0)
    if not bool(torch.isfinite(direction).all()) or float(torch.linalg.vector_norm(direction)) == 0:
        raise ValueError('Zero or nonfinite donor direction')
    return direction


def random_control(direction: torch.Tensor, context_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if direction.ndim != 1 or direction.dtype != torch.float32 or direction.numel() < 2:
        raise ValueError('Invalid direction width/dtype')
    norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(norm)) or float(norm) == 0:
        raise ValueError('Zero or nonfinite direction')
    label = 'voxtral-track-b-answer-link-v3|random-word|' + context_id
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'big') & ((1 << 63) - 1)
    g = torch.Generator(device='cpu').manual_seed(seed)
    d = direction.detach().cpu()
    u = torch.randn(d.shape, generator=g, dtype=torch.float32)
    u = u - torch.dot(u, d) / torch.dot(d, d) * d
    u_norm = torch.linalg.vector_norm(u)
    if not bool(torch.isfinite(u_norm)) or float(u_norm) == 0:
        raise ValueError('Degenerate random control')
    u = (u * (torch.linalg.vector_norm(d) / u_norm)).to(direction.device)
    if abs(float(torch.linalg.vector_norm(u)) / float(norm) - 1) > 1e-5:
        raise ValueError('Random norm mismatch')
    return u, {'seed_label': label, 'seed': seed, 'direction_sha256': tensor_sha(u)}


def freeze_aliases(tokenizer: Any, word: str, *, prompt_ids: Sequence[int],
                   contextual_encode: Callable[[str], Sequence[int]],
                   eos: int = EOS) -> tuple[tuple[int, ...], ...]:
    """Freeze only aliases whose full contextual encoding preserves prompt IDs.

    ``contextual_encode(surface)`` must encode the actual pinned chat prompt
    followed by that assistant surface, without EOS. The runner/preflight owns
    this processor-specific construction; a standalone encode is insufficient.
    """
    if not re.fullmatch(r'[A-Za-z]+', word, re.ASCII):
        raise ValueError('Alias word must be ASCII letters')
    prompt = tuple(int(x) for x in prompt_ids)
    if not prompt:
        raise ValueError('Prompt IDs required for contextual alias validation')
    sequences: set[tuple[int, ...]] = set()
    for fn in ALIAS_CASES:
        for prefix in ('', ' '):
            surface = prefix + fn(word)
            tokens = tuple(int(x) for x in tokenizer.encode(surface, add_special_tokens=False))
            contextual = tuple(int(x) for x in contextual_encode(surface))
            if contextual[:len(prompt)] != prompt or contextual[len(prompt):] != tokens:
                raise ValueError('Alias tokenization differs at actual prompt boundary')
            if not tokens or eos in tokens or any(t < 0 for t in tokens):
                raise ValueError('Invalid alias tokenization')
            sequences.add((*tokens, eos))
    return tuple(sorted(sequences))


def classify(ids: Sequence[int], decoded: str, special_ids: set[int], eos: int,
             recipient: str, donor: str) -> str:
    """Same strict six-way classification as the frozen FIT causal runner."""
    if any(t in special_ids and not (i == len(ids) - 1 and t == eos)
           for i, t in enumerate(ids)):
        return 'malformed_special'
    if not ids or ids[-1] != eos:
        return 'missing_eos'
    surface = decoded.strip()
    if not surface or not re.fullmatch(r'[A-Za-z]+', surface, flags=re.ASCII):
        return 'malformed_surface'
    if surface.casefold() == recipient.casefold():
        return 'recipient'
    if surface.casefold() == donor.casefold():
        return 'opposite'
    return 'other'


@dataclass(frozen=True)
class Edit:
    kind: str  # passive, zero, vector, full
    rows: tuple[int, ...] = ()
    direction: torch.Tensor | None = None
    alpha: float = 0.0
    donor: torch.Tensor | None = None


class ProjectorHook:
    """One-call FP32 hook; Voxtral projector output is flat [B*T,D]."""

    def __init__(self, model: Any, source: torch.Tensor, batch_size: int,
                 placeholders_per_alias: int, edit: Edit, grad: bool = False):
        if (source.ndim != 2 or source.dtype != torch.float32 or
                not bool(torch.isfinite(source).all())):
            raise ValueError('Source must be finite FP32 [T,D]')
        if batch_size < 1 or placeholders_per_alias != source.shape[0]:
            raise ValueError('Batch/placeholder/projector geometry mismatch')
        if edit.kind not in ('passive', 'zero', 'vector', 'full') or (edit.kind == 'passive' and grad):
            raise ValueError('Unknown edit kind')
        if edit.kind == 'vector':
            checked_rows(edit.rows, source.shape[0])
            if (edit.direction is None or edit.direction.shape != (source.shape[1],) or
                    edit.direction.dtype != torch.float32 or
                    not bool(torch.isfinite(edit.direction).all()) or
                    float(torch.linalg.vector_norm(edit.direction)) == 0 or
                    not math.isfinite(edit.alpha) or edit.alpha == 0):
                raise ValueError('Invalid vector edit')
        if edit.kind == 'full' and (
            edit.donor is None or edit.donor.shape != source.shape or
            edit.donor.dtype != torch.float32 or not bool(torch.isfinite(edit.donor).all())
        ):
            raise ValueError('Invalid donor state')
        self.model, self.source, self.batch_size = model, source, batch_size
        self.t, self.width, self.edit, self.grad = *source.shape, edit, grad
        self.calls = 0
        self.leaf: torch.Tensor | None = None
        self.realized: torch.Tensor | None = None
        self.requested_norm: float | None = None
        self.realized_relative_error: float | None = None
        self.pre_hashes: list[str] = []
        self.post_hashes: list[str] = []
        self._handle = None

    def __enter__(self) -> 'ProjectorHook':
        module = self.model.model.multi_modal_projector
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            raise ValueError('Projector hook handle missing')
        self._handle.remove()
        if exc_type is None and self.calls != 1:
            raise ValueError(f'Expected exactly one projector hook call, got {self.calls}')

    def _hook(self, _module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls != 1:
            raise ValueError('Multiple projector calls in one forward')
        if (not isinstance(output, torch.Tensor) or
                output.shape != (self.batch_size * self.t, self.width) or
                output.dtype != torch.float32 or not bool(torch.isfinite(output).all())):
            raise ValueError('Flat projector shape/dtype/finiteness mismatch')
        flat = output.detach()
        segments = flat.reshape(self.batch_size, self.t, self.width)
        for b in range(self.batch_size):
            if not exact_equal(segments[b], self.source):
                raise ValueError(f'Alias {b} prepatch projector differs from singleton source')
            self.pre_hashes.append(tensor_sha(segments[b]))
        if self.edit.kind == 'passive':
            self.realized = torch.zeros_like(segments)
            self.post_hashes = list(self.pre_hashes)
            return output
        leaf = output.detach().clone().requires_grad_(self.grad)
        self.leaf = leaf
        patched = leaf.clone().reshape(self.batch_size, self.t, self.width)
        if self.edit.kind == 'vector':
            if self.edit.direction is None:
                raise ValueError('Vector direction missing')
            rows = list(self.edit.rows)
            d = self.edit.direction.to(device=patched.device)
            patched[:, rows, :] = patched[:, rows, :] + self.edit.alpha * d
        elif self.edit.kind == 'full':
            if self.edit.donor is None:
                raise ValueError('Full donor state missing')
            patched[:, :, :] = self.edit.donor.to(device=patched.device)
        realized = patched - leaf.reshape(self.batch_size, self.t, self.width)
        self.realized = realized
        if self.edit.kind == 'zero' and not exact_equal(patched, leaf.reshape(self.batch_size, self.t, self.width)):
            raise ValueError('Exact-zero hook changed source bits')
        if self.edit.kind == 'vector':
            outside = torch.ones(self.t, dtype=torch.bool, device=patched.device)
            outside[list(self.edit.rows)] = False
            if not exact_equal(patched[:, outside], leaf.reshape(self.batch_size, self.t, self.width)[:, outside]):
                raise ValueError('Vector edit altered outside rows')
            requested = self.edit.alpha * self.edit.direction.to(patched.device)
            requested = requested.expand(self.batch_size, len(self.edit.rows), self.width)
            self.requested_norm = float(torch.linalg.vector_norm(requested))
            rel = float(torch.linalg.vector_norm(realized[:, list(self.edit.rows)] - requested) /
                        torch.linalg.vector_norm(requested))
            self.realized_relative_error = rel
            if not math.isfinite(rel) or rel > 0.05:
                raise ValueError('Realized FP32 edit differs by over 5%')
        for b in range(self.batch_size):
            self.post_hashes.append(tensor_sha(patched[b]))
        return patched.reshape(self.batch_size * self.t, self.width)

    def evidence(self) -> dict[str, Any]:
        if self.calls != 1 or self.realized is None:
            raise ValueError('Hook evidence incomplete')
        return {
            'hook_calls': self.calls, 'flat_shape': [self.batch_size * self.t, self.width],
            'pre_hashes': self.pre_hashes, 'post_hashes': self.post_hashes,
            'realized_sha256': tensor_sha(self.realized),
            'realized_fp32_l2': float(torch.linalg.vector_norm(self.realized.detach().float())),
            'per_alias_realized_fp32_l2': [float(torch.linalg.vector_norm(row.detach().float()))
                                         for row in self.realized],
            'requested_fp32_l2': self.requested_norm,
            'realized_relative_error': self.realized_relative_error,
        }


def _batch_inputs(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    required = {'input_ids', 'attention_mask', 'input_features'}
    if set(batch) != required:
        raise ValueError('Unexpected processor batch fields')
    out = {k: v.to(device) for k, v in batch.items()}
    if (out['input_ids'].ndim != 2 or out['input_ids'].shape[0] != 1 or
            out['attention_mask'].shape != out['input_ids'].shape or
            out['input_features'].shape[0] != 1 or
            not bool((out['attention_mask'] == 1).all())):
        raise ValueError('Expected one unpadded prompt/audio batch')
    return out


def _placeholder_count(ids: torch.Tensor, audio_token: int) -> int:
    return int((ids == audio_token).sum().item())


def capture_projector(model: Any, batch: Mapping[str, torch.Tensor],
                      *, audio_token: int = 24) -> torch.Tensor:
    """One passive source forward; no checkpoint loading or resource admission."""
    inputs = _batch_inputs(batch, batch['input_features'].device)
    t = _placeholder_count(inputs['input_ids'], audio_token)
    found: list[torch.Tensor] = []

    def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
        if (output.shape[0] != t or output.ndim != 2 or
                output.dtype != torch.float32 or not bool(torch.isfinite(output).all())):
            raise ValueError('Passive projector geometry/dtype differs')
        found.append(output.detach().clone())

    handle = model.model.multi_modal_projector.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    if len(found) != 1:
        raise ValueError('Expected exactly one passive projector call')
    return found[0]


def score_alias_class(model: Any, batch: Mapping[str, torch.Tensor],
                      aliases: Sequence[Sequence[int]], source: torch.Tensor,
                      edit: Edit | None, *, pad: int, audio_token: int = 24,
                      grad: bool = False) -> dict[str, Any]:
    """Sequential B=1, EOS-inclusive class mass and its exact weighted gradient.

    The `pad` parameter is checked for interface parity with generation but no
    padding is placed into model inputs. Each alias runs its own bounded reply
    forward: L alias tokens including EOS -> prompt + L-1 input tokens -> L
    output rows, starting at the prompt-final position. This is the V5-safe
    scoring topology; no B=6 memory assumption is made.
    """
    if (not aliases or len({tuple(a) for a in aliases}) != len(aliases) or
            pad < 0 or (grad and edit is None) or
            any(not a or a[-1] != EOS or
                any(not isinstance(t, int) or t < 0 for t in a)
                for a in aliases)):
        raise ValueError('Invalid frozen EOS-inclusive aliases or grad edit')
    inputs = _batch_inputs(batch, source.device)
    prompt_len = inputs['input_ids'].shape[1]
    t = _placeholder_count(inputs['input_ids'], audio_token)
    per_alias: list[float] = []
    per_token: list[list[float]] = []
    per_gradient: list[torch.Tensor] = []
    hooks: list[dict[str, Any] | None] = []
    input_shapes: list[list[int]] = []
    logits_shapes: list[list[int]] = []
    for alias in aliases:
        length = len(alias)
        continuation = torch.tensor([alias[:-1]], dtype=inputs['input_ids'].dtype,
                                    device=inputs['input_ids'].device)
        ids = torch.cat([inputs['input_ids'], continuation], dim=1)
        mask = torch.cat([inputs['attention_mask'], torch.ones_like(continuation)], dim=1)
        if ids.shape != (1, prompt_len + length - 1) or mask.shape != ids.shape:
            raise ValueError('Alias input shape mismatch')
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            with ProjectorHook(model, source, 1, t, edit or Edit('passive'), grad=grad) as hook:
                result = model(input_ids=ids, attention_mask=mask,
                               input_features=inputs['input_features'],
                               use_cache=False, logits_to_keep=length)
                hooked = hook
            logits = result.logits
            if logits.ndim != 3 or logits.shape[:2] != (1, length):
                raise ValueError('Answer-logit slice must be [1,L,V]')
            targets = torch.tensor(alias, dtype=torch.long, device=logits.device)
            token_lp = torch.log_softmax(logits[0].float(), dim=-1).gather(
                1, targets[:, None]).squeeze(1)
            ll = token_lp.sum()
            if grad:
                if hooked.leaf is None:
                    raise ValueError('Differentiable projector leaf missing')
                gradient, = torch.autograd.grad(ll, hooked.leaf)
                if gradient.shape != source.shape or not bool(torch.isfinite(gradient).all()):
                    raise ValueError('Nonfinite or wrong-shape alias source gradient')
                per_gradient.append(gradient.detach())
            per_alias.append(float(ll.detach()))
            per_token.append([float(v.detach()) for v in token_lp])
            hooks.append(hooked.evidence())
            input_shapes.append(list(ids.shape))
            logits_shapes.append(list(logits.shape))
    values = torch.tensor(per_alias, dtype=torch.float64)
    mass = torch.logsumexp(values, dim=0)
    weights = torch.softmax(values, dim=0)
    source_grad = None
    if grad:
        source_grad = sum((float(w) * g for w, g in zip(weights, per_gradient)),
                          torch.zeros_like(source))
        if not bool(torch.isfinite(source_grad).all()):
            raise ValueError('Nonfinite class-mass source gradient')
    return {
        'aliases_with_eos': [list(a) for a in aliases],
        'token_logprobs_including_eos': per_token,
        'alias_logprobs': per_alias, 'alias_softmax_weights': weights.tolist(),
        'log_mass': float(mass), 'gradient': source_grad,
        'alias_lengths_including_eos': [len(a) for a in aliases],
        'full_input_shapes': input_shapes, 'answer_logits_shapes': logits_shapes,
        'hooks': hooks, 'forward_count': len(aliases),
    }


def answer_jacobian(model: Any, batch: Mapping[str, torch.Tensor],
                    recipient_aliases: Sequence[Sequence[int]],
                    donor_aliases: Sequence[Sequence[int]], source: torch.Tensor,
                    direction: torch.Tensor, rows: Sequence[int],
                    *, pad: int, audio_token: int = 24) -> dict[str, Any]:
    fixed = checked_rows(rows, source.shape[0])
    zero = Edit('zero')
    r = score_alias_class(model, batch, recipient_aliases, source, zero,
                          pad=pad, audio_token=audio_token, grad=True)
    d = score_alias_class(model, batch, donor_aliases, source, zero,
                          pad=pad, audio_token=audio_token, grad=True)
    if r['gradient'] is None or d['gradient'] is None:
        raise ValueError('Class-mass gradients missing')
    grad_s = d['gradient'] - r['gradient']
    j = (grad_s[list(fixed)] * direction.to(grad_s.device)).sum()
    if not bool(torch.isfinite(j)):
        raise ValueError('Nonfinite answer Jacobian contraction')
    return {'s0': d['log_mass'] - r['log_mass'], 'j': float(j),
            'gradient_sha256': tensor_sha(grad_s),
            'recipient': r, 'donor': d}


def generate_answer(model: Any, processor: Any, batch: Mapping[str, torch.Tensor],
                    recipient: str, donor_word: str, *, source: torch.Tensor | None,
                    edit: Edit | None, pad: int, audio_token: int = 24,
                    recipient_aliases: Sequence[Sequence[int]] = (),
                    donor_aliases: Sequence[Sequence[int]] = ()) -> dict[str, Any]:
    """Greedy complete suffix. No hook only for the unpatched clean arm."""
    inputs = _batch_inputs(batch, source.device if source is not None else batch['input_features'].device)
    t = _placeholder_count(inputs['input_ids'], audio_token)
    if (source is None) != (edit is None):
        raise ValueError('Source/edit must be supplied together')
    with torch.no_grad():
        if edit is None:
            output = model.generate(**inputs, do_sample=False, use_cache=True,
                                    max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS,
                                    pad_token_id=pad)
            evidence = None
        else:
            with ProjectorHook(model, source, 1, t, edit) as hooked:
                output = model.generate(**inputs, do_sample=False, use_cache=True,
                                        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS,
                                        pad_token_id=pad)
                evidence = hooked.evidence()
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] <= inputs['input_ids'].shape[1]:
        raise ValueError('Invalid complete generation tensor')
    suffix = output[0, inputs['input_ids'].shape[1]:].tolist()
    if len(suffix) > MAX_NEW_TOKENS:
        raise ValueError('Generated suffix exceeds frozen cap')
    decoded = processor.batch_decode([suffix], skip_special_tokens=True)[0]
    with_special = processor.batch_decode([suffix], skip_special_tokens=False)[0]
    label = classify(suffix, decoded, set(processor.tokenizer.all_special_ids),
                     EOS, recipient, donor_word)
    return {'suffix_ids': suffix, 'decoded': decoded, 'decoded_with_special': with_special,
            'class': label, 'terminal_eos': bool(suffix[-1] == EOS),
            'recipient_alias_covered': tuple(suffix) in {tuple(a) for a in recipient_aliases},
            'donor_alias_covered': tuple(suffix) in {tuple(a) for a in donor_aliases},
            'hook': evidence}


def predicted_switch_clauses(clean: Mapping[str, Any], zero: Mapping[str, Any],
                             edited: Mapping[str, Any], *, s0: float, j: float,
                             alpha: float, measured_s: float) -> dict[str, bool]:
    """Frozen V3 answer-link gate; returns every clause instead of a soft label."""
    if not all(math.isfinite(x) for x in (s0, j, alpha, measured_s)) or alpha not in (-0.1, 0.1):
        raise ValueError('Predicted-switch gate needs finite frozen local alpha')
    clauses = {
        'clean_recipient_covered': clean['class'] == 'recipient' and
                                   clean['terminal_eos'] and clean['recipient_alias_covered'],
        'zero_recipient_covered': zero['class'] == 'recipient' and
                                  zero['terminal_eos'] and zero['recipient_alias_covered'],
        'clean_zero_exact_suffix_parity': clean['suffix_ids'] == zero['suffix_ids'],
        's0_negative': s0 < 0,
        'predicted_margin_positive': s0 + alpha * j > 0,
        'measured_margin_positive': measured_s > 0,
        'edited_donor_covered': edited['class'] == 'opposite' and
                                edited['terminal_eos'] and edited['donor_alias_covered'],
    }
    clauses['predicted_complete_answer_switch'] = all(clauses.values())
    return clauses


def assert_score_parity(grad_score: Mapping[str, Any],
                        no_grad_score: Mapping[str, Any], tolerance: float = 1e-4) -> None:
    """Check one class's per-alias grad/no-grad identity before margin claims."""
    if grad_score['aliases_with_eos'] != no_grad_score['aliases_with_eos']:
        raise ValueError('Score parity aliases differ')
    if len(grad_score['hooks']) != len(no_grad_score['hooks']):
        raise ValueError('Score parity forward count differs')
    for left, right, grad_hook, no_grad_hook in zip(
        grad_score['token_logprobs_including_eos'],
        no_grad_score['token_logprobs_including_eos'],
        grad_score['hooks'], no_grad_score['hooks'],
    ):
        if len(left) != len(right) or any(abs(a - b) > tolerance for a, b in zip(left, right)):
            raise ValueError('Grad/no-grad token likelihood parity failed')
        if grad_hook is None or no_grad_hook is None or (
            grad_hook['pre_hashes'] != no_grad_hook['pre_hashes'] or
            grad_hook['post_hashes'] != no_grad_hook['post_hashes']
        ):
            raise ValueError('Grad/no-grad projector state parity failed')
    if abs(grad_score['log_mass'] - no_grad_score['log_mass']) > tolerance:
        raise ValueError('Grad/no-grad class mass parity failed')


def assert_generation_controls(clean: Mapping[str, Any], zero: Mapping[str, Any],
                               natural_donor: Mapping[str, Any],
                               full_donor: Mapping[str, Any]) -> None:
    """Exact complete-suffix parity; a class-only match is insufficient."""
    if clean['suffix_ids'] != zero['suffix_ids']:
        raise ValueError('Exact-zero complete suffix differs from unhooked clean')
    if natural_donor['suffix_ids'] != full_donor['suffix_ids']:
        raise ValueError('Full-projector donor suffix differs from natural donor')
    hook = zero['hook']
    if hook is None or hook['pre_hashes'] != hook['post_hashes'] or hook['realized_fp32_l2'] != 0:
        raise ValueError('Exact-zero source state changed')
