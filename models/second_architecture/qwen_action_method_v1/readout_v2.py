"""No-weight, target-restricted Jacobian readout and fixed comparison baselines.

Inputs are saved FP32/FP64 states, three complete-label log scores, and VJPs.
This does not compute model activations or fit a general 2048x2048 lens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .endpoint import LABELS

LABEL_TOKEN_IDS = (22852, 7384, 88493)
EOS_TOKEN_ID = 151645
FIT_CONTEXTS = 18


def _matrix(value: np.ndarray, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} shape invalid")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} nonfinite")
    return array


def _log_probability(logits: np.ndarray, token_id: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 <= token_id < len(values):
        raise ValueError("invalid vocabulary logits or token ID")
    shifted = values - np.max(values)
    return float(shifted[token_id] - np.log(np.exp(shifted).sum()))


def complete_label_log_score(
    first_step_logits: np.ndarray, after_label_logits: np.ndarray,
    label_token_id: int, *, eos_token_id: int = EOS_TOKEN_ID
) -> float:
    """Log P(one pinned label token, then EOS), not a first-token margin."""
    if label_token_id not in LABEL_TOKEN_IDS or eos_token_id != EOS_TOKEN_ID:
        raise ValueError("unfrozen label/EOS token")
    return (_log_probability(first_step_logits, label_token_id)
            + _log_probability(after_label_logits, eos_token_id))


@dataclass(frozen=True)
class MeanVJP:
    mean_state: np.ndarray
    mean_scores: np.ndarray
    mean_gradients: np.ndarray

    def predict(self, states: np.ndarray) -> np.ndarray:
        value = _matrix(states, (self.mean_state.size,), "evaluation states")
        return self.mean_scores[None, :] + (value - self.mean_state) @ self.mean_gradients.T


def fit_mean_vjp(
    states: np.ndarray, scores: np.ndarray, gradients: np.ndarray,
    *, expected_fit: int = FIT_CONTEXTS
) -> MeanVJP:
    """FIT-only first-order Taylor readout, with exactly three scalar targets."""
    h = np.asarray(states, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != expected_fit or h.shape[1] < 1 or not np.isfinite(h).all():
        raise ValueError("FIT state census/shape invalid")
    y = _matrix(scores, (len(LABELS),), "FIT scores")
    g = np.asarray(gradients, dtype=np.float64)
    if (y.shape[0] != expected_fit or g.shape != (expected_fit, len(LABELS), h.shape[1])
            or not np.isfinite(g).all()):
        raise ValueError("FIT gradient census/shape invalid")
    if not np.any(np.linalg.norm(g, axis=2) > 0):
        raise ValueError("all FIT target gradients are zero")
    return MeanVJP(h.mean(axis=0), y.mean(axis=0), g.mean(axis=0))


@dataclass(frozen=True)
class DualRidge:
    mean_state: np.ndarray
    mean_scores: np.ndarray
    fit_centered: np.ndarray
    dual_coefficients: np.ndarray
    regularization: float

    def predict(self, states: np.ndarray) -> np.ndarray:
        value = _matrix(states, (self.mean_state.size,), "evaluation states")
        kernel = ((value - self.mean_state) @ self.fit_centered.T) / self.mean_state.size
        return self.mean_scores[None, :] + kernel @ self.dual_coefficients


def fit_fixed_ridge(
    states: np.ndarray, scores: np.ndarray, *, expected_fit: int = FIT_CONTEXTS
) -> DualRidge:
    """Fixed dual ridge baseline; lambda = mean training Gram diagonal."""
    h = np.asarray(states, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != expected_fit or h.shape[1] < 1 or not np.isfinite(h).all():
        raise ValueError("FIT state census/shape invalid")
    y = _matrix(scores, (len(LABELS),), "FIT scores")
    if y.shape[0] != expected_fit:
        raise ValueError("FIT score census invalid")
    h_mean, y_mean = h.mean(axis=0), y.mean(axis=0)
    x = h - h_mean
    k = (x @ x.T) / h.shape[1]
    lam = float(np.trace(k) / expected_fit)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("degenerate ridge Gram matrix")
    dual = np.linalg.solve(k + lam * np.eye(expected_fit), y - y_mean)
    return DualRidge(h_mean, y_mean, x, dual, lam)


def constant_score_baseline(fit_scores: np.ndarray, count: int) -> np.ndarray:
    scores = _matrix(fit_scores, (len(LABELS),), "FIT scores")
    if count < 1:
        raise ValueError("empty comparison")
    return np.repeat(scores.mean(axis=0, keepdims=True), count, axis=0)


def predicted_labels(scores: np.ndarray) -> list[str]:
    values = _matrix(scores, (len(LABELS),), "readout scores")
    return [LABELS[int(index)] for index in np.argmax(values, axis=1)]


def target_direction(readout: MeanVJP, target: str, reference: str) -> np.ndarray:
    """Unit target-minus-reference full-sequence score VJP direction."""
    if target not in LABELS or reference not in LABELS or target == reference:
        raise ValueError("invalid class contrast")
    delta = (readout.mean_gradients[LABELS.index(target)]
             - readout.mean_gradients[LABELS.index(reference)])
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("degenerate class direction")
    return delta / norm


def matched_random_direction(unit_target: np.ndarray, seed: int) -> np.ndarray:
    """Independent, norm-matched random control orthogonal to the target."""
    target = np.asarray(unit_target, dtype=np.float64)
    if (target.ndim != 1 or target.size < 2 or not np.isfinite(target).all()
            or not np.isclose(np.linalg.norm(target), 1.0, atol=1e-8)):
        raise ValueError("target must be a finite unit vector")
    rng = np.random.Generator(np.random.PCG64(seed))
    draw = rng.standard_normal(target.size)
    orthogonal = draw - np.dot(draw, target) * target
    norm = float(np.linalg.norm(orthogonal))
    if norm <= 1e-12:
        raise ValueError("random direction degenerate")
    return orthogonal / norm


def development_gate(
    intended: list[str], typed_complete_labels: list[str | None],
    audio_complete_labels: list[str | None], typed_readout: list[str],
    audio_readout: list[str], constant_readout: list[str],
    typed_canonical_paths: list[bool], audio_canonical_paths: list[bool],
    typed_actual_scores: np.ndarray, typed_vjp_scores: np.ndarray,
    typed_constant_scores: np.ndarray, audio_actual_scores: np.ndarray,
    audio_vjp_scores: np.ndarray, audio_constant_scores: np.ndarray,
) -> dict[str, int | float | bool]:
    """Fixed six-script gate; both modalities use the identical script order."""
    rows = (intended, typed_complete_labels, audio_complete_labels,
            typed_readout, audio_readout, constant_readout,
            typed_canonical_paths, audio_canonical_paths)
    if any(len(row) != 6 for row in rows) or any(intended.count(x) != 2 for x in LABELS):
        raise ValueError("development source census changed")
    if any(x not in LABELS for row in (typed_readout, audio_readout, constant_readout)
           for x in row):
        raise ValueError("invalid readout class")
    if len(set(constant_readout)) != 1:
        raise ValueError("constant comparator changed across development items")
    if any(type(value) is not bool for row in (typed_canonical_paths, audio_canonical_paths)
           for value in row):
        raise ValueError("invalid canonical token-path flag")
    counts = {
        "typed_complete": sum(a == b for a, b in zip(intended, typed_complete_labels)),
        "audio_complete": sum(a == b for a, b in zip(intended, audio_complete_labels)),
        "typed_vjp": sum(a == b for a, b in zip(intended, typed_readout)),
        "audio_vjp": sum(a == b for a, b in zip(intended, audio_readout)),
        "constant": sum(a == b for a, b in zip(intended, constant_readout)),
        "typed_canonical": sum(typed_canonical_paths),
        "audio_canonical": sum(audio_canonical_paths),
    }
    for prefix, actual, predicted, baseline in (
        ("typed", typed_actual_scores, typed_vjp_scores, typed_constant_scores),
        ("audio", audio_actual_scores, audio_vjp_scores, audio_constant_scores),
    ):
        matrices = [_matrix(value, (len(LABELS),), prefix + " score")
                    for value in (actual, predicted, baseline)]
        if any(value.shape[0] != 6 for value in matrices):
            raise ValueError("development score census changed")
        zero_mae = float(np.mean(np.abs(matrices[0] - matrices[2])))
        if zero_mae <= 0:
            raise ValueError("degenerate constant-score comparator")
        counts[prefix + "_vjp_to_constant_mae"] = (
            float(np.mean(np.abs(matrices[0] - matrices[1]))) / zero_mae)
    counts["pass"] = (counts["typed_complete"] >= 5
                      and counts["audio_complete"] >= 5
                      and counts["typed_vjp"] >= 5
                      and counts["audio_vjp"] >= 4
                      and counts["typed_vjp"] > counts["constant"]
                      and counts["audio_vjp"] > counts["constant"]
                      and counts["typed_canonical"] >= 5
                      and counts["audio_canonical"] >= 5
                      and counts["typed_vjp_to_constant_mae"] <= 0.9
                      and counts["audio_vjp_to_constant_mae"] <= 0.9)
    return counts
