"""uii_observer.py — UII v19.0 — external observation tooling.

The math spine is descriptive. Φ, ∇Φ, Hessian, vol_opt, C_local,
C_global describe the trajectory the closure loop produces; they are
computed against the substrate's emitted state (sensing channels,
compression's causal graph, coherence's consistency check) by a reader
external to the loop.

This module is not imported by the substrate (uii_triad, uii_operators,
uii_reality, uii_ledger). Every quantity here is computed
from substrate emission. Σ_P is built directly from compression's
causal_graph and active_channel_state. The substrate contains no
scoring inside it — coherence's commit decision is structural pattern-
match, not optimization. Math spine quantities have no return path
into operator update logic; if they ever did, the math would have
stopped being descriptive and become the substrate's measurement of
itself, and its predictive content would be gone.

Usage (in external tooling):

    from uii_observer import PhiField, StateTrace, eigen_decompose
    phi = PhiField()
    val  = phi.phi(state, peak_snapshot=...)
    grad = phi.gradient(state, peak_snapshot=...)
    H, ev, evec, channels, H_C, H_O = phi.compute_hessian(state, ...)
    vol_opt = float(np.sum(ev[ev > 0]))
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from uii_operators import (
    CausalEdge,
    CoherenceOperator,
    CompressionOperator,
    PredictionOperator,
    SensingChannel,
    SensingOperator,
)


def eigen_decompose(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition with regularization fallback."""
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        H = H + 1e-3 * np.eye(H.shape[0])
        eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.clip(eigvals, -1e6, 1e6)
    return eigvals, eigvecs


def _build_sigma_p(compression: CompressionOperator,
                   sensing:     SensingOperator) -> Tuple[np.ndarray, List[str]]:
    """Σ_P over active channels, built from compression's edges and
    sensing's coverage. Diagonal entries are coverage; off-diagonal
    entries accumulate edge.weight × edge.confidence × source_coverage.
    Symmetrized before return.

    Pure function of substrate emission — does not modify operator
    state and is safe to call from external tooling.
    """
    channels = sensing.channels
    active   = [cid for cid, ch in channels.items() if ch.active]
    n        = len(active)
    if n == 0:
        return np.zeros((0, 0)), []
    idx = {cid: i for i, cid in enumerate(active)}
    sigma_p = np.zeros((n, n))
    for i, cid in enumerate(active):
        sigma_p[i, i] = channels[cid].coverage
    for (src, tgt), edge in compression.causal_graph.items():
        if src in idx and tgt in idx:
            sigma_p[idx[src], idx[tgt]] += (
                edge.weight * edge.confidence * channels[src].coverage
            )
    sigma_p = (sigma_p + sigma_p.T) / 2.0
    return sigma_p, active


class StateTrace:
    """External per-iteration record of c_local; running mean is c_global.

    C_local is the cosine alignment between the math spine's gradient
    ∇Φ and the substrate's actual trajectory ẋ (read as last_delta on
    each channel). C_global is the running mean of c_local. Computed
    from outside; not consumed by the loop. C_local positive on average
    is one of the cleanest empirical signals that composition is
    producing the dynamics the math spine describes.
    """

    def __init__(self, max_length: int = 1000):
        self.history:        deque = deque(maxlen=max_length)
        self.c_local_history: deque = deque(maxlen=100)
        self.c_global:       float = 1.0
        self._last_gradient: Dict[str, float] = {}

    def compute_c_local(self,
                         gradient: Dict[str, float],
                         sensing:  SensingOperator) -> Optional[float]:
        x_dot  = {cid: ch.last_delta for cid, ch in sensing.channels.items()
                  if ch.active and cid in gradient}
        common = set(gradient.keys()) & set(x_dot.keys())
        if not common:
            return None
        g_vec = np.array([gradient[k] for k in common])
        x_vec = np.array([x_dot[k]    for k in common])
        ng, nx = np.linalg.norm(g_vec), np.linalg.norm(x_vec)
        if ng < 1e-8 or nx < 1e-8:
            return None
        return float(np.dot(g_vec, x_vec) / (ng * nx))

    def record(self,
               state_dict: Dict[str, float],
               sensing:    Optional[SensingOperator]    = None,
               gradient:   Optional[Dict[str, float]]   = None,
               virtual:    bool                         = False):
        self.history.append(state_dict)
        if gradient is not None:
            self._last_gradient = gradient
        if not virtual and self._last_gradient and sensing is not None:
            c_local = self.compute_c_local(self._last_gradient, sensing)
            if c_local is not None:
                self.c_local_history.append(c_local)
                self.c_global = float(np.mean(self.c_local_history))

    def get_recent(self, n: int) -> List[Dict]:
        if len(self.history) < n:
            return list(self.history)
        return list(self.history)[-n:]

    def __len__(self) -> int:
        return len(self.history)


class PhiField:
    """Computes Φ(x) = α·C + β·log(O) + γ·K and its gradient/Hessian.

    Φ describes the trajectory the closure loop produces. Computed
    against the substrate's emitted operator state by an external
    observer. The substrate never reads Φ. C is the coherence
    component (edge weight × confidence × source_coverage × tgt_coverage,
    summed and normalized), O is the optionality component (sum of
    positive eigenvalues of Σ_P — the reachable future volume), K is
    the coverage-distance-from-peak component (negative L2 distance
    from a peak snapshot's coverage vector).

    Relation to the math spine: vol_opt = sum(positive eigenvalues of
    Σ_P). The Hessian of Φ has three blocks: H_C from compression's
    edges (weight × confidence² × cov × cov), H_O from Σ_P⁻¹
    (regularized to positive definite), H_K from the coverage delta
    outer product. The substrate's commit gate does not consult these
    quantities; it consults coherence.commit_decision, which is
    structural.
    """

    O_FLOOR: float = 1e-6

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def _compute_C(self, state) -> float:
        graph = state.compression.causal_graph
        if not graph:
            return 0.0
        total = 0.0
        for (src, tgt), edge in graph.items():
            src_cov = state.sensing.channels.get(
                src, SensingChannel(src, False, 0, 0, 0)).coverage
            tgt_cov = state.sensing.channels.get(
                tgt, SensingChannel(tgt, False, 0, 0, 0)).coverage
            total += abs(edge.weight) * edge.confidence * src_cov * tgt_cov
        max_possible = 2.0 * len(graph)
        return float(np.clip(total / max(max_possible, 1.0), 0.0, 1.0))

    def _compute_O(self, state) -> float:
        sigma_p, active = _build_sigma_p(state.compression, state.sensing)
        if len(active) == 0:
            return self.O_FLOOR
        eigenvalues, _ = np.linalg.eigh(sigma_p)
        vol_opt = float(np.sum(eigenvalues[eigenvalues > 0]))
        return float(max(vol_opt, self.O_FLOOR))

    def _compute_K(self, state, peak_snapshot: Optional[Dict] = None) -> float:
        if peak_snapshot is None:
            return 0.0

        peak_channels = peak_snapshot.get('sensing', {}).get('channels', {})
        if not peak_channels:
            return 0.0

        active = [cid for cid, ch in state.sensing.channels.items() if ch.active]
        if not active:
            return 0.0

        current_cov = np.array([state.sensing.channels[cid].coverage
                                 for cid in active])
        peak_cov    = np.array([float(peak_channels.get(cid, {}).get('coverage', 0.0))
                                 for cid in active])

        return -float(np.sum((current_cov - peak_cov) ** 2))

    def phi(self, state, peak_snapshot: Optional[Dict] = None) -> float:
        C = self._compute_C(state)
        O = self._compute_O(state)
        K = self._compute_K(state, peak_snapshot)
        phi_val = (self.alpha * C
                   + self.beta * np.log(max(O, self.O_FLOOR))
                   + self.gamma * K)
        return float(np.clip(phi_val, -100.0, 100.0))

    def gradient(self, state,
                 peak_snapshot: Optional[Dict] = None) -> Dict[str, float]:
        channels = state.sensing.channels
        grad: Dict[str, float] = {}

        O = self._compute_O(state)

        peak_coverage: Dict[str, float] = {}
        if peak_snapshot is not None:
            peak_ch_snap = peak_snapshot.get('sensing', {}).get('channels', {})
            for cid in channels:
                if cid in peak_ch_snap:
                    peak_coverage[cid] = float(peak_ch_snap[cid].get('coverage', 0.0))

        graph = state.compression.causal_graph

        for cid, ch in channels.items():
            if not ch.active:
                continue

            g_C = 0.0
            for (src, tgt), edge in graph.items():
                if src == cid:
                    tgt_cov = channels.get(
                        tgt, SensingChannel(tgt, False, 0, 0, 0)).coverage
                    g_C += edge.weight * edge.confidence * tgt_cov
                elif tgt == cid:
                    src_cov = channels.get(
                        src, SensingChannel(src, False, 0, 0, 0)).coverage
                    g_C += edge.weight * edge.confidence * src_cov
            max_possible = 2.0 * max(len(graph), 1)
            g_C /= max_possible

            g_O = 0.0
            eps = 1e-4
            try:
                perturbed_channels = dict(channels)
                perturbed_ch = dataclasses.replace(
                    ch, coverage=float(np.clip(ch.coverage + eps, 0.0, 1.0))
                )
                perturbed_channels[cid] = perturbed_ch
                ps_sensing = SensingOperator(channels=perturbed_channels)

                class _PState:
                    pass
                ps = _PState()
                ps.sensing     = ps_sensing
                ps.compression = state.compression
                O_perturbed = self._compute_O(ps)
                g_O = (O_perturbed - O) / eps
            except Exception:
                g_O = 0.0

            g_K = 0.0
            if cid in peak_coverage:
                g_K = -2.0 * (ch.coverage - peak_coverage[cid])

            grad[cid] = (self.alpha * g_C
                         + self.beta / max(O, self.O_FLOOR) * g_O
                         + self.gamma * g_K)

        norm = float(np.sqrt(sum(v ** 2 for v in grad.values())))
        if norm > 1e-8:
            grad = {k: v / norm for k, v in grad.items()}
        return grad

    def compute_hessian(self,
                        state,
                        peak_snapshot: Optional[Dict] = None,
                        epsilon:       float = 1e-4,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   List[str], np.ndarray, np.ndarray]:
        """Hessian of Φ over active channels. Returns (H, eigenvalues,
        eigenvectors, active_channel_ids, H_C, H_O). H_C and H_O are
        returned separately so external tooling can decompose Φ's
        Hessian into its coherence and optionality components.

        Σ_P is built from compression's causal_graph and active channel
        coverage — pure function of substrate emission, no operator
        method calls.
        """
        channels = state.sensing.channels
        active   = [cid for cid, ch in channels.items() if ch.active]
        n        = len(active)
        _zero6 = (np.zeros((0, 0)), np.array([]), np.zeros((0, 0)),
                  [], np.zeros((0, 0)), np.zeros((0, 0)))
        if n == 0:
            return _zero6

        idx = {cid: i for i, cid in enumerate(active)}

        H_C = np.zeros((n, n))
        for (src, tgt), edge in state.compression.causal_graph.items():
            if src in idx and tgt in idx:
                si = idx[src]
                ti = idx[tgt]
                src_cov = channels[src].coverage
                tgt_cov = channels[tgt].coverage
                H_C[si, ti] += edge.weight * (edge.confidence ** 2) * src_cov * tgt_cov
        H_C = (H_C + H_C.T) / 2.0

        sigma_p, _ = _build_sigma_p(state.compression, state.sensing)
        if sigma_p.shape[0] > 0:
            try:
                ev, evec = np.linalg.eigh(sigma_p)
            except np.linalg.LinAlgError:
                sigma_p += 1e-3 * np.eye(sigma_p.shape[0])
                ev, evec = np.linalg.eigh(sigma_p)
            pos = ev > 1e-9
            if np.any(pos):
                inv_ev = np.minimum(1.0 / ev[pos], 1e3)
                H_O = evec[:, pos] @ np.diag(inv_ev) @ evec[:, pos].T
            else:
                H_O = np.zeros((n, n))
        else:
            H_O = np.zeros((n, n))

        if peak_snapshot is not None:
            peak_channels = peak_snapshot.get('sensing', {}).get('channels', {})
            current_cov = np.array([channels[cid].coverage for cid in active])
            peak_cov    = np.array([float(peak_channels.get(cid, {}).get('coverage', 0.0))
                                     for cid in active])
            delta = current_cov - peak_cov
            H_K = -np.outer(delta, delta)
        else:
            H_K = np.zeros((n, n))

        H = (self.alpha * H_C
             + self.beta  * H_O
             + self.gamma * H_K
             + epsilon * np.eye(n))

        ev_H, evec_H = eigen_decompose(H)
        return H, ev_H, evec_H, active, H_C, H_O
