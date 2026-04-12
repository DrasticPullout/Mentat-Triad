"""
uii_geometry.py — v16
Merged foundation and field geometry layer.

Replaces uii_types.py entirely. All imports that previously pointed at
uii_types must be redirected here.

Also absorbs:
  - SymbolGroundingAdapter, SYMBOL_GROUNDING_PROMPT (from uii_intelligence.py)
  - DeathClock / renamed from LatentDeathClock (from uii_coherence.py)

Contents:
  Constants
    BASE_AFFORDANCES, INTERFACE_COUPLED_SIGNALS, POTENTIALLY_INVARIANT_SIGNALS,
    SUBSTRATE_DIMS

  State
    SubstrateState         — operator-backed SIPA geometry (backward compat intact)
    StateTrace             — trajectory + C_local/C_global tracking

  Field geometry
    eigen_decompose()      — module-level: always eigh, clipped eigenvalues
    expected_optionality_gain() — module-level: E[Δlog(O(a))] per action
    PhiField               — v16: _compute_K changed; compute_hessian + score_actions new;
                             all v15.3 methods retained exactly

  Constraint kernel
    CRKEvaluation, CRKVerdict, CRKMonitor — unchanged from v15.3

  Trajectory manifold
    TrajectoryCandidate, TrajectoryManifold — unchanged

  Agent infrastructure
    AgentHandler, UserAgentHandler, AVAILABLE_AGENTS — unchanged

  Adapters (ABCs)
    RealityAdapter

  Orchestration
    DeathClock             — minimal step/token budget tracker (renamed from
                             LatentDeathClock); elaborate counting eliminated —
                             resource pressure sensed through api_llm channel

  Symbol grounding
    SYMBOL_GROUNDING_PROMPT, SymbolGroundingAdapter — moved from uii_intelligence.py;
                             core logic unchanged; SRE coupling removed

What is NOT here
    LAYER1_PARAMS              — dead with velocity/generation system
    VELOCITY_FIELD             — dead with velocity/generation system
    RelationAdapter            — eliminated with SRE
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from uii_operators import (
    CausalEdge,
    ChannelPrediction,
    CoherenceOperator,
    CompressionOperator,
    DEFAULT_CHANNELS,
    OperatorConsistencyCheck,
    PredictionOperator,
    SensingChannel,
    SensingOperator,
    SelfModifyingOperator,
    SMOUpdate,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_AFFORDANCES: Set[str] = {
    'navigate', 'click', 'fill', 'type', 'read',
    'scroll', 'observe', 'delay', 'evaluate',
    'query_agent',
    'python',
    'llm_query',  # symbol grounding affordance — outcome feeds CouplingMatrixEstimator
}

# Interface-coupled signals — artifacts of HTML/browser rendering, not reality structure.
# BLOCKED from becoming discovered ledger axes.
INTERFACE_COUPLED_SIGNALS: Set[str] = {
    'dom_depth', 'element_count', 'link_count', 'button_count',
    'input_count', 'scroll_position', 'viewport_height', 'dom_complexity',
}

# Potentially invariant signals — may survive substrate change.
# ALLOWED to proceed to the full AxisAdmissionTest.
POTENTIALLY_INVARIANT_SIGNALS: Set[str] = {
    'response_latency',    # network reality (not DOM)
    'content_entropy',     # information-theoretic (text/element ratio)
    'surface_change_rate', # how much interactive surface changed (normalized)
    'interaction_density', # interactive fraction of total (normalized)
}

SUBSTRATE_DIMS: List[str] = ['S', 'I', 'P', 'A']


# ──────────────────────────────────────────────────────────────────────────────
# SubstrateState
# ──────────────────────────────────────────────────────────────────────────────

class SubstrateState:
    """
    v15.1+: Four-dimensional information processing geometry backed by DASS operators.

    S, I, P, A are properties derived from operators — all existing code that
    reads state.S, state.I, state.P, state.A continues to work unchanged.
    These scalar properties are for observation and logging only — not decision paths.
    """

    def __init__(self,
                 sensing:     SensingOperator,
                 compression: CompressionOperator,
                 prediction:  PredictionOperator,
                 coherence:   CoherenceOperator):
        self.sensing     = sensing
        self.compression = compression
        self.prediction  = prediction
        self.coherence   = coherence

    # ── Scalar proxies — derived from operators ───────────────────────────────

    @property
    def S(self) -> float:
        return self.sensing.to_scalar_proxy()

    @property
    def I(self) -> float:
        return self.compression.to_scalar_proxy()

    @property
    def P(self) -> float:
        return self.prediction.to_grounded_proxy(self.sensing, self.compression)

    @property
    def A(self) -> float:
        return self.coherence.to_scalar_proxy()

    @A.setter
    def A(self, value: float):
        # Backward compat — A is derived from coherence operator; setter is a no-op.
        pass

    def as_dict(self) -> Dict[str, float]:
        return {"S": self.S, "I": self.I, "P": self.P, "A": self.A}


# ──────────────────────────────────────────────────────────────────────────────
# StateTrace
# ──────────────────────────────────────────────────────────────────────────────

class StateTrace:
    """Ordered history of substrate states for field calculations."""

    def __init__(self, max_length: int = 1000):
        self.history:        deque = deque(maxlen=max_length)
        self.c_local_history: deque = deque(maxlen=100)
        self.c_global:       float = 0.0
        self._last_gradient: Dict[str, float] = {}

    def compute_c_local(self,
                        gradient: Dict[str, float],
                        sensing:  SensingOperator) -> float:
        """
        C_local = ⟨∇Φ, ẋ⟩ / (‖∇Φ‖ · ‖ẋ‖)
        ẋ = last_delta across active channels.
        Returns 0.0 during bootstrap (gradient near-zero) — not gradient-lost.
        """
        x_dot  = {cid: ch.last_delta for cid, ch in sensing.channels.items()
                  if ch.active and cid in gradient}
        common = set(gradient.keys()) & set(x_dot.keys())
        if not common:
            return 0.0
        g_vec = np.array([gradient[k] for k in common])
        x_vec = np.array([x_dot[k]    for k in common])
        ng, nx = np.linalg.norm(g_vec), np.linalg.norm(x_vec)
        if ng < 1e-8 or nx < 1e-8:
            return 0.0
        return float(np.dot(g_vec, x_vec) / (ng * nx))

    @staticmethod
    def compute_c_local_static(gradient: Dict[str, float],
                                sensing:  SensingOperator) -> float:
        """Static version for use in virtual trajectory scoring."""
        x_dot  = {cid: ch.last_delta for cid, ch in sensing.channels.items()
                  if ch.active and cid in gradient}
        common = set(gradient.keys()) & set(x_dot.keys())
        if not common:
            return 0.0
        g_vec = np.array([gradient[k] for k in common])
        x_vec = np.array([x_dot[k]    for k in common])
        ng, nx = np.linalg.norm(g_vec), np.linalg.norm(x_vec)
        if ng < 1e-8 or nx < 1e-8:
            return 0.0
        return float(np.dot(g_vec, x_vec) / (ng * nx))

    def record(self, state: SubstrateState,
               gradient: Dict[str, float] = None,
               virtual: bool = False):
        """
        v15.2+: gradient parameter added. Computes C_local internally every step.
        virtual=True: state enters history but C_local is NOT recorded.
        """
        self.history.append(state.as_dict())
        if gradient is not None:
            self._last_gradient = gradient
        if not virtual and self._last_gradient:
            c_local = self.compute_c_local(self._last_gradient, state.sensing)
            self.c_local_history.append(c_local)
            self.c_global = float(np.mean(self.c_local_history)) \
                if self.c_local_history else 0.0

    def get_recent(self, n: int) -> List[Dict]:
        if len(self.history) < n:
            return list(self.history)
        return list(self.history)[-n:]

    def __len__(self) -> int:
        return len(self.history)


# ──────────────────────────────────────────────────────────────────────────────
# GroundingSpec — v16.1
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundingSpec:
    """
    Geometric context passed from the Triad to SymbolGroundingAdapter.ground_symbol().

    Built once per step from Phase 2 output. Carries the field's intent —
    what channels it needs signal on and in what direction — so the LLM
    receives a precise spec rather than an open-ended prompt.

    The Triad reasons. The LLM fills in the blank.

    Fields
    ------
    desired_delta         : channel_id → natural gradient component.
                            Positive = field wants this channel to grow.
                            Derived from H⁻¹ ∇Φ in active channel space.
    dark_channels         : channels with coverage < 0.3 AND gradient above
                            median. What the field most wants to sense but can't.
    top_gradient_channels : top-5 channels by |∇Φ| regardless of coverage.
                            Used when dark_channels is empty.
    current_url           : browser URL at grounding time.
    page_title            : page title at grounding time.
    nat_grad_magnitude    : ‖H⁻¹ ∇Φ‖ — overall field drive strength.
    """
    desired_delta:          Dict[str, float]
    dark_channels:          List[str]
    top_gradient_channels:  List[str]
    current_url:            str
    page_title:             str
    nat_grad_magnitude:     float = 0.0

    def sensing_target_summary(self, max_channels: int = 5) -> str:
        """Token-minimal summary for LLM prompts."""
        targets = self.dark_channels[:max_channels] or self.top_gradient_channels[:max_channels]
        if not targets:
            return "no specific channel targets"
        parts = []
        for cid in targets:
            delta = self.desired_delta.get(cid, 0.0)
            direction = "↑" if delta > 0 else "↓"
            parts.append(f"{cid}{direction}")
        return ", ".join(parts)

    def desired_delta_summary(self, max_channels: int = 5) -> str:
        """Top desired deltas for LLM prompts. Token-minimal."""
        sorted_by_magnitude = sorted(
            self.desired_delta.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True
        )[:max_channels]
        return ", ".join(f"{cid}={v:+.3f}" for cid, v in sorted_by_magnitude)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level geometry functions
# ──────────────────────────────────────────────────────────────────────────────

def eigen_decompose(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stable symmetric eigendecomposition.

    Always uses eigh (symmetric Hermitian). Never eig.
    Fallback: add 1e-3·I if LinAlgError.
    Clip eigenvalues to [-1e6, 1e6].

    Returns (eigenvalues, eigenvectors) in ascending order (eigh convention).
    """
    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        H = H + 1e-3 * np.eye(H.shape[0])
        eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.clip(eigvals, -1e6, 1e6)
    return eigvals, eigvecs

def expected_optionality_gain(prediction:         PredictionOperator,
                               action:             str,
                               compression:        CompressionOperator,
                               sensing:            SensingOperator,
                               coupling_estimator  = None) -> float:
    """
    E[Δlog(O(a))] = log_vol(Σ_P_post) - log_vol(Σ_P_pre)

    log_vol counts only positive eigenvalues — near-zero eigenvalues
    (dark or barely-active channels) do not penalize actions that open
    new surface. Consistent with Vol_opt definition.

    Forming path: when causal graph mean confidence < 0.1, covariance
    EOG is unreliable. Use empirical affordance deltas if available,
    else return uniform small positive to keep all actions viable.

    Called once per step for all actions before viable set construction.
    Result shared between C2 and score_actions(). Never recomputed inside either.
    """
    # Forming: causal graph not yet reliable enough for covariance EOG.
    if compression.causal_graph:
        mean_conf = float(np.mean([e.confidence
                                    for e in compression.causal_graph.values()]))
    else:
        mean_conf = 0.0

    if mean_conf < 0.1:
        if coupling_estimator is not None:
            deltas = coupling_estimator.affordance_deltas.get(action, [])
            if deltas:
                return float(np.mean([
                    np.mean([abs(v) for v in d.values()])
                    for d in deltas if isinstance(d, dict) and d
                ]))
        return 0.05   # uniform — no data, no ranking, keep all actions viable

    sigma_pre, active = prediction.covariance_matrix(sensing, compression)

    if sigma_pre.shape[0] == 0:
        return 0.0

    delta = prediction.test_virtual(
        compression, action, phi_field=None, sensing=sensing,
        coupling_estimator=coupling_estimator,
    )

    channels = sensing.channels
    idx      = {cid: i for i, cid in enumerate(active)}
    n        = len(active)

    hyp_coverage = {
        cid: float(np.clip(channels[cid].coverage + delta.get(cid, 0.0), 0.0, 1.0))
        for cid in active
    }

    sigma_post = np.zeros((n, n))
    for i, cid in enumerate(active):
        sigma_post[i, i] = hyp_coverage[cid]
    for (src, tgt), edge in compression.causal_graph.items():
        if src in idx and tgt in idx:
            sigma_post[idx[src], idx[tgt]] += (
                edge.weight * edge.confidence * hyp_coverage[src]
            )
    sigma_post = (sigma_post + sigma_post.T) / 2.0

    def _log_vol(sigma: np.ndarray) -> float:
        ev  = np.linalg.eigvalsh(sigma)
        pos = ev[ev > 1e-9]
        if len(pos) == 0:
            return 0.0
        return float(np.sum(np.log(pos)))

    return _log_vol(sigma_post) - _log_vol(sigma_pre)

# ──────────────────────────────────────────────────────────────────────────────
# PhiField — v16
# ──────────────────────────────────────────────────────────────────────────────

class PhiField:
    """
    Information Potential Field (Φ-field) — v16.

    Three-term potential over operator space:
        Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)

    v16 changes vs v15.3:
      _compute_K  — now operates in active channel space only;
                    no longer uses to_coupling_matrix() (4×4 SIPA distance).
                    peak_snapshot is now ledger.operator_snapshot format.
      compute_hessian — NEW: H = α·H_C + β·H_O + γ·H_K + εI in channel space.
      score_actions   — NEW: maturity × nat_grad_align + (1-maturity) × E[Δlog(O)].

    All v15.3 methods retained exactly:
      _compute_C, _build_sigma_p, _direction_crk_viable, _compute_O,
      phi, gradient.

    The gradient() K-term reads peak_snapshot using the v16 operator_snapshot
    format. When 'sensing.channels' key is absent it returns 0.0 gracefully —
    which is correct because H_K in compute_hessian() carries the attractor
    well contribution through the Hessian, so the gradient K term is secondary.
    """

    O_FLOOR = 1e-6

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma

    # ── C(x) — unchanged from v15.3 ──────────────────────────────────────────

    def _compute_C(self, state) -> float:
        """
        C(x) = Σ_{edges} |w(e)|·conf(e)·cov(src)·cov(tgt)  /  (2·|edges|)
        Unchanged from v15.3.
        """
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

    # ── Σ_P builder — unchanged from v15.3 ───────────────────────────────────

    def _build_sigma_p(self, state) -> Tuple[np.ndarray, List[str]]:
        """
        Build Σ_P over active channels from causal graph.
        Diagonal: Σ_P[i,i] = coverage_i
        Off-diagonal: Σ_P[i,j] = w(i→j) × conf(i→j) × coverage_i
        Returns (sigma_p, active_channel_list). sigma_p is symmetrized.
        Unchanged from v15.3.
        """
        channels = state.sensing.channels
        active = [cid for cid, ch in channels.items() if ch.active]
        n = len(active)
        if n == 0:
            return np.zeros((0, 0)), []

        idx = {cid: i for i, cid in enumerate(active)}
        sigma_p = np.zeros((n, n))

        for i, cid in enumerate(active):
            sigma_p[i, i] = channels[cid].coverage

        for (src, tgt), edge in state.compression.causal_graph.items():
            if src in idx and tgt in idx:
                sigma_p[idx[src], idx[tgt]] += (
                    edge.weight * edge.confidence * channels[src].coverage
                )

        sigma_p = (sigma_p + sigma_p.T) / 2.0
        return sigma_p, active

    # ── CRK viability filter — unchanged from v15.3 ──────────────────────────

    def _direction_crk_viable(self, channel_delta: Dict[str, float], state) -> bool:
        """
        Check whether moving in a channel direction is CRK-viable.
        Forming guard: empty causal_graph → all directions viable.
        Unchanged from v15.3.
        """
        if not state.compression.causal_graph:
            return True

        graph_channels: Set[str] = set()
        for (src, tgt) in state.compression.causal_graph:
            graph_channels.add(src)
            graph_channels.add(tgt)

        total_delta = sum(abs(v) for v in channel_delta.values())
        if total_delta < 1e-8:
            return True

        ungrounded = sum(abs(v) for cid, v in channel_delta.items()
                         if cid not in graph_channels)
        if ungrounded / total_delta > 0.7:
            return False

        channel_weights: Dict[str, float] = {}
        for (src, tgt), edge in state.compression.causal_graph.items():
            w = abs(edge.weight) * edge.confidence
            channel_weights[src] = channel_weights.get(src, 0.0) + w
            channel_weights[tgt] = channel_weights.get(tgt, 0.0) + w

        if channel_weights:
            max_w     = max(channel_weights.values())
            threshold = max_w * 0.5
            anchors   = {cid for cid, w in channel_weights.items() if w >= threshold}
            anchor_reduction = sum(
                -v for cid, v in channel_delta.items()
                if cid in anchors and v < 0
            )
            if anchor_reduction / total_delta > 0.5:
                return False

        return True

    # ── O(x) — unchanged from v15.3 ──────────────────────────────────────────

    def _compute_O(self, state) -> float:
        """
        O(x) = Σ_{λ_i > 0} λ_i  of  Σ_P filtered to CRK-viable directions.
        Unchanged from v15.3.
        """
        sigma_p, active = self._build_sigma_p(state)
        if len(active) == 0:
            return self.O_FLOOR

        eigenvalues, eigenvectors = np.linalg.eigh(sigma_p)

        vol_opt = 0.0
        for k, lam in enumerate(eigenvalues):
            if lam <= 0:
                continue
            v = eigenvectors[:, k]
            channel_delta = {active[j]: float(v[j]) for j in range(len(active))}
            if self._direction_crk_viable(channel_delta, state):
                vol_opt += lam

        return float(max(vol_opt, self.O_FLOOR))

    # ── K(x) — CHANGED in v16 ────────────────────────────────────────────────

    def _compute_K(self, state, peak_snapshot: Optional[Dict] = None) -> float:
        """
        K(x) = -‖coverage_current - coverage_peak‖²

        coverage_current: coverage values of active channels.
        coverage_peak:    from ledger operator_snapshot['sensing']['channels'].

        CHANGED from v15.3:
          - No longer reads to_coupling_matrix() (4×4 SIPA distance removed).
          - Operates in active channel space only.
          - peak_snapshot is the full ledger.operator_snapshot dict.

        If peak_snapshot is None: K = 0.0 (cold start).
        The compression geometry is fully captured by H_C in the Hessian.
        """
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

    # ── Φ(x) — unchanged from v15.3 ──────────────────────────────────────────

    def phi(self, state, trace, peak_snapshot: Optional[Dict] = None) -> float:
        """
        Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)
        peak_snapshot: ledger.operator_snapshot or None.
        Unchanged from v15.3 (K now reads new snapshot format).
        """
        C = self._compute_C(state)
        O = self._compute_O(state)
        K = self._compute_K(state, peak_snapshot)
        phi_val = (self.alpha * C
                   + self.beta * np.log(max(O, self.O_FLOOR))
                   + self.gamma * K)
        return float(np.clip(phi_val, -100.0, 100.0))

    # ── ∇Φ — unchanged from v15.3 ────────────────────────────────────────────

    def gradient(self, state, trace,
                 peak_snapshot: Optional[Dict] = None) -> Dict[str, float]:
        """
        ∇Φ(x) per channel, normalized to unit vector.

        Three contributions:
          ∂C/∂channel_i  — closed form
          ∂O/∂channel_i  — finite difference
          ∂K/∂channel_i  = -2·(cov_i - cov_peak_i)

        peak_snapshot format: ledger.operator_snapshot.
        The K-term reads ['sensing']['channels'][cid]['coverage'].
        Falls back gracefully to 0.0 when key absent (cold start).

        trace parameter accepted but not used in computation —
        safe to pass None (as score_actions() does).

        Unchanged from v15.3 modulo K-term snapshot key update.
        """
        channels = state.sensing.channels
        grad: Dict[str, float] = {}

        O = self._compute_O(state)

        # Build peak coverage dict from v16 operator_snapshot format
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

            # ∂C/∂channel_i (closed form)
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

            # ∂O/∂channel_i (finite difference)
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

            # ∂K/∂channel_i = -2·(cov_i - cov_peak_i)
            g_K = 0.0
            if cid in peak_coverage:
                g_K = -2.0 * (ch.coverage - peak_coverage[cid])

            grad[cid] = (self.alpha * g_C
                         + self.beta / max(O, self.O_FLOOR) * g_O
                         + self.gamma * g_K)

        # Normalize to unit vector
        norm = float(np.sqrt(sum(v ** 2 for v in grad.values())))
        if norm > 1e-8:
            grad = {k: v / norm for k, v in grad.items()}
        return grad

    # ── compute_hessian — NEW in v16 ─────────────────────────────────────────

    def compute_hessian(self,
                        state,
                        prediction:    PredictionOperator,
                        coherence:     CoherenceOperator,
                        peak_snapshot: Optional[Dict] = None,
                        epsilon:       float = 1e-4,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   List[str], np.ndarray, np.ndarray]:
        """
        H = α·H_C + β·H_O + γ·H_K + εI

        All three components in active channel space (n×n where n = active
        channel count). Same space as Φ, ∇Φ, and Σ_P. Not the 4×4 SIPA space.

        H_C — compression curvature:
            weight * confidence² * coverage_src * coverage_tgt per causal edge,
            symmetrized. Confidence² damping prevents low-confidence early edges
            from exploding eigenvalues.

        H_O — optionality curvature (inverse covariance):
            Σ_P = covariance_matrix(). Decompose: eigvecs, eigvals = eigh(Σ_P).
            H_O = eigvecs[:, pos] @ diag(1/eigvals[pos]) @ eigvecs[:, pos].T
            Naturally caps Hessian eigenvalue magnitude.
            If no positive eigenvalues: H_O = 0.

        H_K — attractor well:
            delta = current_configuration_vector() - peak_configuration_vector()
            H_K = -outer(delta, delta)
            If peak_snapshot is None: H_K = 0.

        εI — diagonal regularization (prevents singular matrices).

        Returns (H, eigenvalues, eigenvectors, active_channel_list, H_C, H_O).
        H_C and H_O returned so score_actions() can compute eigenspectrum
        maturity ratio without recomputing.

        Called ONCE per step. Caller caches the return tuple; it is NOT
        recomputed inside score_actions(), viable set loop, or anywhere else.

        Returns (zeros(0,0), [], [], [], zeros(0,0), zeros(0,0)) if no
        active channels.
        """
        channels = state.sensing.channels
        active   = [cid for cid, ch in channels.items() if ch.active]
        n        = len(active)
        _zero6   = (np.zeros((0, 0)), np.array([]), np.zeros((0, 0)),
                    [], np.zeros((0, 0)), np.zeros((0, 0)))
        if n == 0:
            return _zero6

        idx = {cid: i for i, cid in enumerate(active)}

        # ── H_C ───────────────────────────────────────────────────────────────
        H_C = np.zeros((n, n))
        for (src, tgt), edge in state.compression.causal_graph.items():
            if src in idx and tgt in idx:
                si  = idx[src]
                ti  = idx[tgt]
                src_cov = channels[src].coverage
                tgt_cov = channels[tgt].coverage
                H_C[si, ti] += edge.weight * (edge.confidence ** 2) * src_cov * tgt_cov
        H_C = (H_C + H_C.T) / 2.0

        # ── H_O ───────────────────────────────────────────────────────────────
        sigma_p, _ = prediction.covariance_matrix(state.sensing, state.compression)
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

        # ── H_K ───────────────────────────────────────────────────────────────
        if peak_snapshot is not None:
            delta = (
                coherence.current_configuration_vector(state.sensing, active)
                - coherence.peak_configuration_vector(peak_snapshot, active)
            )
            H_K = -np.outer(delta, delta)
        else:
            H_K = np.zeros((n, n))

        # ── Combine ───────────────────────────────────────────────────────────
        H = (self.alpha * H_C
             + self.beta  * H_O
             + self.gamma * H_K
             + epsilon * np.eye(n))

        ev_H, evec_H = eigen_decompose(H)

        return H, ev_H, evec_H, active, H_C, H_O

    # ── score_actions — NEW in v16 ────────────────────────────────────────────

    def score_actions(self,
                      viable_actions:        List[str],
                      state,
                      H:                     np.ndarray,
                      eigvals:               np.ndarray,
                      eigvecs:               np.ndarray,
                      active_channels:       List[str],
                      H_C:                   np.ndarray,
                      H_O:                   np.ndarray,
                      prediction:            PredictionOperator,
                      peak_snapshot:         Optional[Dict] = None,
                      optionality_gain_dict: Optional[Dict[str, float]] = None,
                      coupling_estimator     = None,
                      ) -> Dict[str, float]:
        """
        score(a) = maturity × nat_grad_align_norm(a) + (1 - maturity) × E[Δlog(O(a))]

        Dynamical law: ẋ = G⁻¹(x) ∇Φ(x) where G(x) = H = ∇²Φ locally.
        Lyapunov: dΦ/dt = ∇Φᵀ G⁻¹ ∇Φ ≥ 0 (G⁻¹ PD when H PSD).
        Φ is guaranteed non-decreasing. Ascent is information-optimal.

        nat_grad_align(a):
            natural_gradient = H⁻¹ ∇Φ
            δx(a) = test_virtual() channel delta in active channel space
            nat_grad_align(a) = ⟨natural_gradient, δx(a)⟩

        H⁻¹ via eigendecomposition (already cached — NOT recomputed):
            H⁻¹ = eigvecs @ diag(1/eigvals_safe) @ eigvecs.T
            eigvals clipped at 1e-6 to avoid inversion instability.

        maturity = var_H_C / (var_H_C + var_H_O + ε)
            Derived from Hessian eigenspectrum. No proxy.
            H_C dominates → causal graph has real structure → δ²Φ trustworthy → 1.
            H_O dominates → prediction uncertainty high → explore → 0.

        E[Δlog(O(a))] pre-computed once per step by caller; passed via
        optionality_gain_dict OR recomputed here if not pre-provided.

        Both nat_grad_align and E[Δlog(O)] are normalized to [0, 1] before
        combining so maturity is a true blend weight.

        H, eigvals, eigvecs, H_C, H_O: from cached compute_hessian().
        Do NOT recompute inside this method.

        Returns {action: score} dict.
        Empty viable_actions or empty channel space returns {a: 0.0} for each action.
        """
        if not viable_actions:
            return {}
        if H.shape[0] == 0:
            return {a: 0.0 for a in viable_actions}

        # ── Maturity from eigenspectrum ratio ─────────────────────────────────
        ev_C   = np.linalg.eigvalsh(self.alpha * H_C)
        ev_O   = np.linalg.eigvalsh(self.beta  * H_O)
        var_C  = float(np.sum(ev_C[ev_C > 0]))
        var_O  = float(np.sum(ev_O[ev_O > 0]))
        maturity = var_C / (var_C + var_O + 1e-8)

        # ── Natural gradient: H⁻¹ ∇Φ ─────────────────────────────────────────
        grad_dict = self.gradient(state, None, peak_snapshot)
        grad_vec  = np.array([grad_dict.get(cid, 0.0) for cid in active_channels])

        eigvals_safe = np.where(np.abs(eigvals) > 1e-6,
                                eigvals,
                                1e-6 * np.sign(eigvals + 1e-12))
        H_inv            = eigvecs @ np.diag(1.0 / eigvals_safe) @ eigvecs.T
        natural_gradient = H_inv @ grad_vec

        channel_idx = {cid: i for i, cid in enumerate(active_channels)}
        n           = len(active_channels)

        nat_grad_align_raw: Dict[str, float] = {}
        optionality_gain:   Dict[str, float] = {}

        for action in viable_actions:
            # Action direction in active channel space
            channel_delta = state.prediction.test_virtual(
                state.compression, action, phi_field=None, sensing=state.sensing,
                coupling_estimator=coupling_estimator,
            )
            dx = np.zeros(n)
            for cid, val in channel_delta.items():
                if cid in channel_idx:
                    dx[channel_idx[cid]] = val

            nat_grad_align_raw[action] = float(np.dot(natural_gradient, dx))

            # Use pre-computed E[Δlog(O)] if provided (spec: computed once per step)
            if optionality_gain_dict is not None and action in optionality_gain_dict:
                optionality_gain[action] = optionality_gain_dict[action]
            else:
                optionality_gain[action] = expected_optionality_gain(
                    prediction, action, state.compression, state.sensing,
                    coupling_estimator=coupling_estimator,
                )

        # ── Normalize nat_grad_align to [0, 1] ───────────────────────────────
        vals    = list(nat_grad_align_raw.values())
        v_min   = min(vals)
        v_range = max(vals) - v_min
        if v_range < 1e-10:
            nat_norm = {a: 0.5 for a in viable_actions}
        else:
            nat_norm = {a: (nat_grad_align_raw[a] - v_min) / v_range
                        for a in viable_actions}

        # ── Normalize E[Δlog(O)] to [0, 1] ───────────────────────────────────
        og_vals  = list(optionality_gain.values())
        og_min   = min(og_vals)
        og_range = max(og_vals) - og_min
        if og_range < 1e-10:
            og_norm = {a: 0.5 for a in viable_actions}
        else:
            og_norm = {a: (optionality_gain[a] - og_min) / og_range
                       for a in viable_actions}

        return {
            a: maturity * nat_norm[a] + (1.0 - maturity) * og_norm[a]
            for a in viable_actions
        }


# ──────────────────────────────────────────────────────────────────────────────
# CRK structures — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CRKEvaluation:
    constraint:  str    # 'C1'–'C7'
    phase:       str    # 'pre_action' | 'post_action'
    status:      str    # 'satisfied' | 'degraded' | 'violated'
    risk:        float  # [0, 1]
    attribution: str    # 'internal' | 'external' | 'mixed'
    blocks:      bool
    signal:      str


@dataclass
class CRKVerdict:
    phase:         str
    action:        str
    evaluations:   List[CRKEvaluation]
    coherent:      bool
    repair:        Optional[str]
    phi_modifier:  float
    smo_permitted: bool


class CRKMonitor:
    """
    Constraint Recognition Kernel (CRK) — v16 geometric gating.

    Pre-action:  filters action manifold via field geometry; returns phi_modifier.
    Post-action: gates SMO update; failed → rollback signal.

    C1 and C2 pre-action no longer use static action type lists.
    All actions are evaluated geometrically via E[Δlog(O(a))].
    Reversibility is a property of outcome, not type — the Hessian answers it.
    """

    PREDICTION_ERROR_THRESHOLD = 0.05   # retained for any legacy references

    # C3: accurate causal attribution
    C3_A_DROP_THRESHOLD    = 0.05   # coherence drop that warrants checking attribution
    C3_S_STABLE_THRESHOLD  = 0.03   # |ΔS| below this → environment wasn't the cause
    C3_P_ERROR_THRESHOLD   = 0.10   # prediction error above this → internal model failed

    # C4: overconfidence without reality contact
    C4_OVERCONFIDENCE_THRESHOLD = 0.5
    C4_SIGNAL_THRESHOLD         = 0.05

    def evaluate_pre_action(self,
                             proposed_action:    str,
                             coherence:          CoherenceOperator,
                             sensing:            SensingOperator,
                             compression:        CompressionOperator,
                             prediction:         PredictionOperator,
                             field_state:        Dict,
                             coupling_estimator  = None,
                             ) -> CRKVerdict:
        crk_sig   = coherence.crk_signal()
        loop_cl   = crk_sig['loop_closure']
        sig_dev   = crk_sig['signature_deviation']
        i_p       = crk_sig['i_p_consistency']
        evaluations: List[CRKEvaluation] = []

        # Pre-compute E[Δlog(O)] once — used by both C1 and C2.
        # forming: Hessian not yet reliable — eog computation degrades gracefully
        # but block decisions during forming are permissive to allow bootstrap.
        if compression.causal_graph:
            mean_conf = float(np.mean([e.confidence
                                       for e in compression.causal_graph.values()]))
        else:
            mean_conf = 0.0
        forming = mean_conf < 0.1

        eog = 0.0
        if not forming:
            try:
                eog = expected_optionality_gain(
                    prediction, proposed_action, compression, sensing
                )
            except Exception:
                eog = 0.0

        # ── C1: Continuity ────────────────────────────────────────────────────
        # Block when the action threatens the manifold and the loop is already
        # fragile. Geometric criterion: significant eigenvolume collapse combined
        # with low loop closure. No static action type list — the geometry answers
        # whether this action is safe to take given current coherence.
        #
        # Thresholds:
        #   eog < -0.1  : meaningful collapse, not noise
        #   loop_cl < 0.3 : loop already fragile — cannot absorb further loss
        #
        # During forming: C1 is permissive (eog unreliable, bootstrap must explore).
        c1_stability_risk = 0.0
        try:
            # Use empirical SIPA coupling matrix from CouplingMatrixEstimator.
            # That matrix is built from observed state_before/state_after on every
            # execute() — it measures how SIPA dims actually co-move.
            # to_coupling_matrix() (channel→SIPA projection via CHANNEL_TO_DIM)
            # is removed: channels are S's domain; I/P/A do not receive raw channels.
            if coupling_estimator is not None:
                coupling = coupling_estimator.matrix
            else:
                coupling = np.eye(4)   # identity fallback during bootstrap
            pred_delta = self._get_predicted_delta(proposed_action, compression)
            delta_vec  = np.array([pred_delta.get(d, 0.0) for d in ['S', 'I', 'P', 'A']])
            stability  = float(0.5 * delta_vec @ coupling @ delta_vec)
            c1_stability_risk = float(np.clip(-stability, 0.0, 1.0)) if stability < -0.1 else 0.0
        except Exception:
            pass

        if forming:
            c1_blocks = False
        else:
            c1_blocks = (eog < -0.1) and (loop_cl < 0.3)

        base_c1_risk = float(np.clip(1.0 - loop_cl + sig_dev + c1_stability_risk * 0.3, 0.0, 1.0))
        evaluations.append(CRKEvaluation(
            constraint  = 'C1',
            phase       = 'pre_action',
            status      = 'violated' if c1_blocks else ('degraded' if loop_cl < 0.5 else 'satisfied'),
            risk        = base_c1_risk,
            attribution = 'internal',
            blocks      = c1_blocks,
            signal      = (f'loop_closure={loop_cl:.2f}, sig_dev={sig_dev:.2f}, '
                           f'eog={eog:.4f}, stability_risk={c1_stability_risk:.3f}'),
        ))

        # ── C2: Optionality ───────────────────────────────────────────────────
        # Every action is evaluated geometrically. During forming the Hessian is
        # not yet reliable so C2 is permissive. After forming: block any action
        # that strictly collapses positive eigenvolume (eog < -0.01 with margin).
        # No static committing/non-committing distinction — the geometry applies
        # to all actions equally.
        if forming:
            c2_blocks = False
        else:
            c2_blocks = eog < -0.01

        evaluations.append(CRKEvaluation(
            constraint  = 'C2',
            phase       = 'pre_action',
            status      = 'satisfied' if forming else (
                          'violated' if c2_blocks else (
                          'degraded' if eog < 0.0 else 'satisfied')),
            risk        = 0.0 if forming else float(np.clip(-eog, 0.0, 1.0)),
            attribution = 'internal',
            blocks      = c2_blocks,
            signal      = (f'forming=True mean_conf={mean_conf:.3f}'
                           if forming else
                           f'eog={eog:.4f}'),
        ))

        # ── C3: Non-Internalization ───────────────────────────────────────────
        c3_blocks = i_p < 0.3
        evaluations.append(CRKEvaluation(
            constraint  = 'C3',
            phase       = 'pre_action',
            status      = 'violated' if c3_blocks else ('degraded' if i_p < 0.5 else 'satisfied'),
            risk        = float(np.clip(0.5 - i_p, 0.0, 1.0)),
            attribution = 'internal',
            blocks      = c3_blocks,
            signal      = f'i_p_consistency={i_p:.2f}',
        ))

        # ── C6: Agenthood / resource ──────────────────────────────────────────
        system_load = field_state.get('system_load', 0.0)
        evaluations.append(CRKEvaluation(
            constraint  = 'C6',
            phase       = 'pre_action',
            status      = 'degraded' if system_load > 0.7 else 'satisfied',
            risk        = float(np.clip(system_load - 0.7, 0.0, 1.0)) if system_load > 0.7 else 0.0,
            attribution = 'external',
            blocks      = False,
            signal      = f'system_load={system_load:.2f}',
        ))

        # ── C7: Coherence — migrate gate ──────────────────────────────────────
        # migrate is the one action where type identity matters architecturally:
        # it exits the current M entirely to find a substrate where M can be
        # re-established. Requiring loop_cl >= 0.5 before migrate ensures the
        # current loop is coherent enough to serialize and transfer.
        phi_trend = field_state.get('phi_trend', 0.0)
        c7_blocks = proposed_action == 'migrate' and loop_cl < 0.5
        evaluations.append(CRKEvaluation(
            constraint  = 'C7',
            phase       = 'pre_action',
            status      = 'violated' if c7_blocks else ('degraded' if phi_trend < -0.1 else 'satisfied'),
            risk        = float(np.clip(0.5 - loop_cl, 0.0, 1.0)) if c7_blocks else 0.0,
            attribution = 'mixed',
            blocks      = c7_blocks,
            signal      = f'migrate+loop_cl={loop_cl:.2f}' if c7_blocks else f'phi_trend={phi_trend:.2f}',
        ))

        phi_modifier = float(np.clip(float(np.prod([
            1.0 - (e.risk * (1.0 if e.status == 'degraded' else 2.0))
            for e in evaluations
        ])), 0.0, 1.0))
        coherent = not any(e.blocks for e in evaluations)

        repair   = None
        violated = [e for e in evaluations if e.status == 'violated']
        if violated:
            repair = {'C1': 'stabilize', 'C2': 'expand',
                      'C3': 'reattribute', 'C7': 'stabilize'}.get(violated[0].constraint)
        elif any(e.constraint == 'C6' and e.status == 'degraded' for e in evaluations):
            repair = 'coordinate'

        return CRKVerdict(
            phase         = 'pre_action',
            action        = proposed_action,
            evaluations   = evaluations,
            coherent      = coherent,
            repair        = repair,
            phi_modifier  = phi_modifier,
            smo_permitted = True,
        )

    def _get_predicted_delta(self, action: str,
                              compression: CompressionOperator) -> Dict[str, float]:
        action_map = getattr(compression, 'action_substrate_map', {}) or {}
        if action in action_map:
            return dict(action_map[action])
        fallback = {
            'navigate': {'S': 0.05, 'I': 0.0, 'P': -0.08, 'A': 0.0},
            'click':    {'S': 0.02, 'I': 0.0, 'P': -0.02, 'A': 0.0},
            'read':     {'S': 0.03, 'I': 0.0, 'P':  0.0,  'A': 0.0},
            'observe':  {'S': 0.0,  'I': 0.0, 'P':  0.0,  'A': 0.0},
        }
        return fallback.get(action, {'S': 0.0, 'I': 0.0, 'P': 0.0, 'A': 0.0})

    def evaluate_post_action(self,
                              proposed_smo_update: Dict,
                              prior_sensing:       SensingOperator,
                              new_sensing:         SensingOperator,
                              prior_coherence:     CoherenceOperator,
                              new_coherence:       CoherenceOperator,
                              compression:         CompressionOperator,
                              prediction:          PredictionOperator,
                              prior_compression:   CompressionOperator,
                              prediction_errors:   Dict,
                              smo_plasticity:      float = 0.5,
                              ) -> CRKVerdict:
        evaluations: List[CRKEvaluation] = []
        evaluations.append(self._c4_post(prediction, new_sensing, compression))
        evaluations.append(self._c5_post(new_coherence))
        evaluations.append(self._c1_post(proposed_smo_update, prior_compression, compression))
        evaluations.append(self._c3_post(
            prior_sensing, new_sensing,
            prior_coherence, new_coherence,
            prediction_errors,
        ))

        smo_permitted = not any(e.blocks for e in evaluations)
        repair = None
        if not smo_permitted:
            c4 = next((e for e in evaluations if e.constraint == 'C4' and e.blocks), None)
            c1 = next((e for e in evaluations if e.constraint == 'C1' and e.blocks), None)
            c5 = next((e for e in evaluations if e.constraint == 'C5'
                       and e.status in ('degraded', 'violated')), None)
            if c4:
                repair = None
            elif c1:
                repair = 'rollback'
            elif c5:
                repair = 'reattribute'
        if repair is None:
            c3v = next((e for e in evaluations
                        if e.constraint == 'C3' and e.status == 'violated'), None)
            if c3v:
                repair = 'reattribute'

        return CRKVerdict(
            phase         = 'post_action',
            action        = str(proposed_smo_update),
            evaluations   = evaluations,
            coherent      = smo_permitted,
            repair        = repair,
            phi_modifier  = 1.0,
            smo_permitted = smo_permitted,
        )

    def _c3_post(self,
                 prior_sensing:   SensingOperator,
                 new_sensing:     SensingOperator,
                 prior_coherence: CoherenceOperator,
                 new_coherence:   CoherenceOperator,
                 prediction_errors: Dict) -> CRKEvaluation:
        """
        C3: Non-Internalization — accurate causal attribution.

        Spec: Δ(outcome_value) must not imply Δ(self_authority) < 0.

        Violation: A dropped (loop_closure fell) while S was stable
        AND prediction errors were large — meaning the system is blaming
        itself (coherence collapsing) for an internal model failure in
        a stable environment. Correct response: P updates accuracy, A recovers.

        All signals derived from operator state — no SIPA delta dict.
        """
        delta_A   = (new_coherence.consistency.loop_closure
                     - prior_coherence.consistency.loop_closure)
        a_dropped = delta_A < -self.C3_A_DROP_THRESHOLD

        if not a_dropped:
            return CRKEvaluation('C3', 'post_action', 'satisfied', 0.0, 'external', False,
                                 f'ΔA={delta_A:+.3f} — coherence stable')

        # Was the environment the cause? — S moved significantly
        delta_S   = abs(new_sensing.to_scalar_proxy() - prior_sensing.to_scalar_proxy())
        env_moved = delta_S > self.C3_S_STABLE_THRESHOLD

        if env_moved:
            return CRKEvaluation('C3', 'post_action', 'satisfied', abs(delta_A), 'external', False,
                                 f'ΔA={delta_A:+.3f}, ΔS={delta_S:.3f} — '
                                 f'environment drove the drop, attribution correct')

        # Was it an internal model failure?
        mean_error   = (float(np.mean(list(prediction_errors.values())))
                        if prediction_errors else 0.0)
        model_failed = mean_error > self.C3_P_ERROR_THRESHOLD

        if model_failed:
            return CRKEvaluation('C3', 'post_action', 'violated', abs(delta_A),
                                 'internal', False,
                                 f'ΔA={delta_A:+.3f}, ΔS={delta_S:.3f}, '
                                 f'P_error={mean_error:.3f} — '
                                 f'coherence collapsed on internal model failure; '
                                 f'P should learn, A should recover')

        return CRKEvaluation('C3', 'post_action', 'satisfied', abs(delta_A), 'mixed', False,
                             f'ΔA={delta_A:+.3f}, ΔS={delta_S:.3f}, '
                             f'P_error={mean_error:.3f} — cause ambiguous, monitoring')

    def _c4_post(self,
                 prediction:  PredictionOperator,
                 sensing:     SensingOperator,
                 compression: CompressionOperator) -> CRKEvaluation:
        """
        C4: Reality Constraint — model field as independent, uncertain.

        Spec: uncertainty(field_model) > 0.
        Violation = deterministic overconfidence without reality contact.

        High realized_horizon (P thinks it can predict far ahead)
        AND low mean signal_rate across active channels (S not getting
        fresh signal from Reality to ground those predictions).

        Reads directly from operator state — no SIPA delta dict.
        """
        if len(compression.causal_graph) == 0:
            return CRKEvaluation('C4', 'post_action', 'satisfied', 0.0, 'external', False,
                                 'forming — no model yet')

        realized_horizon = prediction.realized_horizon
        horizon_norm     = float(np.clip(realized_horizon / 50.0, 0.0, 1.0))

        if horizon_norm < 1e-6:
            return CRKEvaluation('C4', 'post_action', 'satisfied', 0.0, 'external', False,
                                 'horizon=0 — bootstrap, no overconfidence possible')

        active_channels = [ch for ch in sensing.channels.values() if ch.active]
        if not active_channels:
            return CRKEvaluation('C4', 'post_action', 'satisfied', 0.0, 'external', False,
                                 'no active channels')

        mean_signal    = float(np.mean([ch.signal_rate for ch in active_channels]))
        signal_norm    = float(np.clip(mean_signal / self.C4_SIGNAL_THRESHOLD, 0.0, 1.0))
        overconfidence = horizon_norm * (1.0 - signal_norm)

        if overconfidence > self.C4_OVERCONFIDENCE_THRESHOLD:
            return CRKEvaluation('C4', 'post_action', 'violated',
                                 overconfidence, 'internal', False,
                                 f'horizon={realized_horizon} steps, '
                                 f'signal_rate={mean_signal:.4f} — '
                                 f'predicting without opening the door')

        return CRKEvaluation('C4', 'post_action', 'satisfied', overconfidence, 'external', False,
                             f'horizon={realized_horizon}, signal={mean_signal:.4f} — grounded')

    def _c1_post(self, proposed_update: Dict,
                 prior_compression: CompressionOperator,
                 new_compression:   CompressionOperator) -> CRKEvaluation:
        prior_edges     = set(prior_compression.causal_graph.keys())
        new_edges       = set(new_compression.causal_graph.keys())
        edges_destroyed = prior_edges - new_edges
        if edges_destroyed:
            ratio = len(edges_destroyed) / max(len(prior_edges), 1)
            if ratio > 0.3:
                return CRKEvaluation('C1', 'post_action', 'violated', ratio, 'internal', True,
                                     f'{len(edges_destroyed)} edges destroyed '
                                     f'({ratio:.1%}) — historical erasure, roll back')
        confidence_loss = [
            key for key in prior_edges & new_edges
            if prior_compression.causal_graph[key].confidence > 0.5
            and new_compression.causal_graph[key].confidence < 0.1
        ]
        if len(confidence_loss) > 3:
            return CRKEvaluation('C1', 'post_action', 'degraded',
                                 len(confidence_loss) / max(len(prior_edges), 1),
                                 'internal', False,
                                 f'{len(confidence_loss)} high-confidence edges collapsed')
        return CRKEvaluation('C1', 'post_action', 'satisfied', 0.0, 'internal', False,
                             'continuity preserved')

    def _c5_post(self, coherence: CoherenceOperator) -> CRKEvaluation:
        i_p = coherence.consistency.i_p_consistency
        p_a = coherence.consistency.p_a_consistency
        if i_p < 0.4 and p_a > 0.6:
            return CRKEvaluation('C5', 'post_action', 'degraded', 1.0 - i_p, 'internal', False,
                                 f'i_p={i_p:.2f} low, p_a={p_a:.2f} fine — '
                                 f'error source is internal compression, not external Reality. '
                                 f'Reattribute before SMO update.')
        return CRKEvaluation('C5', 'post_action', 'satisfied', 0.0, 'external', False,
                             'attribution clear')

# ──────────────────────────────────────────────────────────────────────────────
# Trajectory manifold — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryCandidate:
    """Multi-step executable procedure with structural annotations."""
    steps:                            List[Dict]
    rationale:                        str
    estimated_coherence_preservation: float
    estimated_optionality_delta:      float
    reversibility_point:              int
    tested:                           bool           = False
    test_phi_final:                   Optional[float] = None
    test_state_final:                 Optional[Dict]  = None
    test_violations:                  Optional[List]  = None
    test_perturbation_trace:          Optional[List]  = None
    test_succeeded:                   bool            = False
    virtual_phi:                      Optional[float] = None

    def __repr__(self):
        status  = "✓" if self.test_succeeded else "✗" if self.tested else "?"
        phi_str = f"Φ={self.test_phi_final:.3f}" if self.test_phi_final is not None else "untested"
        return f"{status} [{len(self.steps)} steps] {self.rationale[:50]} ({phi_str})"


@dataclass
class TrajectoryManifold:
    """Container for enumerated trajectory space."""
    candidates:          List[TrajectoryCandidate]
    enumeration_context: Dict

    def size(self) -> int:
        return len(self.candidates)

    def tested_count(self) -> int:
        return sum(1 for c in self.candidates if c.tested)

    def get_best(self) -> Optional[TrajectoryCandidate]:
        valid = [c for c in self.candidates if c.tested and c.test_succeeded]
        if not valid:
            return None
        return max(valid, key=lambda c: c.test_phi_final)

    def get_all_tested(self) -> List[TrajectoryCandidate]:
        tested = [c for c in self.candidates if c.tested]
        return sorted(tested,
                      key=lambda c: c.test_phi_final if c.test_phi_final is not None else -1000,
                      reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Agent infrastructure — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

class AgentHandler(ABC):
    """Interface for agent interaction (CRK C6: Other-Agent Existence)."""

    @abstractmethod
    def post_query(self, triad_id: str, query_text: str): ...

    @abstractmethod
    def get_response(self, triad_id: str) -> Optional[str]: ...


class UserAgentHandler(AgentHandler):
    """Human user as agent. Non-blocking query/response."""

    def __init__(self):
        self.pending_queries: deque = deque()
        self.responses: Dict[str, str] = {}

    def post_query(self, triad_id: str, query_text: str):
        self.pending_queries.append({
            'triad_id': triad_id, 'query': query_text, 'timestamp': time.time()
        })
        print(f"\n{'='*70}")
        print(f"[QUERY FROM TRIAD {triad_id}]")
        print(f"{query_text}")
        print(f"{'='*70}")
        print(f"Respond with: triad.respond_to_query('{triad_id}', 'your answer')")
        print(f"Or leave pending - Triad will continue exploration")
        print(f"{'='*70}\n")

    def get_response(self, triad_id: str) -> Optional[str]:
        return self.responses.pop(triad_id, None)

    def respond(self, triad_id: str, answer: str):
        self.responses[triad_id] = answer

    def has_pending(self) -> bool:
        return len(self.pending_queries) > 0

    def get_pending_count(self) -> int:
        return len(self.pending_queries)


AVAILABLE_AGENTS: Dict[str, AgentHandler] = {
    'user': UserAgentHandler()
}


# ──────────────────────────────────────────────────────────────────────────────
# Adapter ABCs — unchanged from v15.3
# ──────────────────────────────────────────────────────────────────────────────

class RealityAdapter(ABC):
    """Interface for environment / perturbation source."""

    @abstractmethod
    def execute(self, action: Dict) -> Tuple[Dict[str, float], Dict]: ...

    @abstractmethod
    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]: ...

    @abstractmethod
    def get_current_affordances(self) -> Dict: ...

    @abstractmethod
    def close(self): ...


# ──────────────────────────────────────────────────────────────────────────────
# Symbol grounding — moved from uii_intelligence.py
# Core logic unchanged; SRE/CausalDiagnosis coupling removed.
# ──────────────────────────────────────────────────────────────────────────────

SYMBOL_GROUNDING_PROMPT = """\
You are the symbol grounding layer of an autonomous system.

The system's structural engine has diagnosed an impossibility and determined
what trajectory shapes are needed. Your job is to fill in concrete parameters
— URLs, Python code, CSS selectors — for the shapes below.

CAUSAL DIAGNOSIS:
  Binding dimension: {binding_dim}
  Cause:             {cause_class}
  Evidence:
{evidence_lines}
  Migration indicated: {migration_indicated}
  Symbol required:     {symbol_requirement}
  Migration urgency:   {migration_urgency}

TRAJECTORY SHAPES TO GROUND:
{shapes_block}

CURRENT ENVIRONMENT:
  URL:     {current_url}
  Title:   {page_title}
  Links:   {links}
  Buttons: {buttons}
  Inputs:  {inputs}

BOUNDARY PRESSURE: {boundary_pressure:.2f}
(1.0 = resource limit imminent. High pressure = prioritise migration.)

TOKEN BUDGET REMAINING: {token_budget_remaining}
(If low: write shorter code, single target only, skip verification steps.)

Fill in concrete parameters for each shape.
For python actions: write complete, runnable code.
For navigate actions: provide a specific, reachable URL.
For click/fill/read actions: provide a valid CSS selector.

Output a JSON array of {n_shapes} trajectories in this format:
[
  {{
    "steps": [
      {{"action_type": "...", "parameters": {{...}}}},
      ...
    ],
    "rationale": "what Φ geometry this trajectory probes",
    "estimated_coherence_preservation": 0.XX,
    "estimated_optionality_delta": 0.XX,
    "reversibility_point": N
  }}
]

JSON only. No commentary.
"""


# ── v16.1 per-action grounding prompts ───────────────────────────────────────
# Each prompt is minimal: context in, one token out, no reasoning.
# The Triad has already decided what action to take and what the field needs.
# The LLM's only job is symbol completion.

NAVIGATE_GROUNDING_PROMPT = """\
Return a single URL. Nothing else. No explanation.

Current page: {current_url}
Page title: {page_title}
Field needs signal on: {sensing_targets}
Available page links: {links}

If the available links include one relevant to the sensing targets, return it.
Otherwise return a URL for a page that would provide signal on those channels.
URL only.\
"""

FILL_GROUNDING_PROMPT = """\
Return text to enter in a form field. Nothing else. No explanation.

Current page: {current_url}
Input type: {input_type}
Placeholder: {placeholder}
Field needs signal on: {sensing_targets}

Return the single best text to enter. Text only.\
"""

EVALUATE_GROUNDING_PROMPT = """\
Return a JavaScript expression for Playwright's page.evaluate(). Nothing else.

Current page: {current_url}
Channels needing signal: {dark_channels}
Desired direction: {desired_delta}

The expression must return a value. Called as: await page.evaluate(EXPR)
One expression only. No function wrapper unless necessary. No explanation.\
"""

PYTHON_GROUNDING_PROMPT = """\
Return executable Python code. Nothing else. No explanation.

Current working directory: {cwd}
Desired substrate delta: {desired_delta}
Channels needing signal: {sensing_targets}

Requirements:
- Store the primary result in a variable named 'result'
- Handle exceptions internally — do not let the code raise
- Minimal code only — achieve the delta, nothing more

Code only. No markdown. No explanation.\
"""


def _format_shapes_block(shapes: List) -> str:
    """
    Format shape list for the LLM prompt.
    Accepts shape objects or dicts — both are handled via getattr/get.
    SRE TrajectoryShape coupling removed; works with any shape-like structure.
    """
    lines: List[str] = []
    for i, shape in enumerate(shapes, 1):
        # Support both object attributes and dict keys
        def _get(key: str, default=''):
            if isinstance(shape, dict):
                return shape.get(key, default)
            return getattr(shape, key, default)

        lines.append(f"Shape {i}: {_get('strategy_class', 'unknown')}")
        seq = _get('action_sequence', [])
        lines.append(f"  Sequence:  {' → '.join(seq) if seq else '(none)'}")
        lines.append(f"  Target:    {_get('target_dimension', '')}")
        lines.append(f"  Rationale: {_get('rationale', '')}")
        pred = _get('predicted_delta', {})
        if isinstance(pred, dict):
            delta_str = ', '.join(
                f"{d}={pred.get(d, 0):+.3f}"
                for d in ['S', 'I', 'P', 'A']
                if abs(pred.get(d, 0)) > 0.001
            )
            if delta_str:
                lines.append(f"  Predicted: {delta_str}")
        lines.append("")
    return "\n".join(lines)


class SymbolGroundingAdapter:
    """
    Symbol grounding layer. Moved from uii_intelligence.py — core logic unchanged.

    Receives a diagnosis context dict and fills in concrete symbols —
    URLs, Python code, CSS selectors — using the LLM's pretrained token field.

    In v16 SymbolGroundingAdapter is called directly from step() in uii_triad.py
    when action ∈ {python, llm_query, migrate}. It no longer requires a
    CausalDiagnosis from StructuralRelationEngine (SRE eliminated).

    The diagnosis parameter in ground_trajectories() is now a plain dict with
    the same keys that SYMBOL_GROUNDING_PROMPT expects. Any dict-like context
    containing the required keys works.

    All parsing, validation, and trajectory history logic unchanged from v15.
    """

    def __init__(self, llm_client):
        self.llm                = llm_client
        self.call_count         = 0

    def ground_trajectories(self,
                             diagnosis: Dict,
                             context:   Dict) -> TrajectoryManifold:
        """
        Build minimal prompt from diagnosis dict, call LLM, parse result.
        context provides current affordances for symbol resolution.

        diagnosis keys used:
          binding_dim, cause_class, evidence, migration_indicated,
          symbol_requirement, migration_urgency, trajectory_shapes

        token_budget_remaining included in prompt so LLM can adapt code
        complexity and target count.
        """
        self.call_count += 1

        affordances        = context.get('affordances', {})
        boundary_pressure  = context.get('boundary_pressure', 0.0)
        shapes             = diagnosis.get('trajectory_shapes', [])

        migration_urgency = diagnosis.get('migration_urgency', 'focused')
        if migration_urgency == 'emergency':
            shapes = shapes[:1]

        token_budget_remaining = "unknown"

        evidence       = diagnosis.get('evidence', [])
        evidence_lines = "\n".join(f"    - {e}" for e in evidence)

        links   = json.dumps(affordances.get('links',   [])[:15])
        buttons = json.dumps(affordances.get('buttons', [])[:10])
        inputs  = json.dumps(affordances.get('inputs',  [])[:8])

        is_bootstrap = affordances.get('bootstrap_state', False)
        current_url  = 'about:blank' if is_bootstrap else affordances.get('current_url', '')
        page_title   = '' if is_bootstrap else affordances.get('page_title', '')

        shapes_block = _format_shapes_block(shapes)

        prompt = SYMBOL_GROUNDING_PROMPT.format(
            binding_dim            = diagnosis.get('binding_dim',        'unknown'),
            cause_class            = diagnosis.get('cause_class',        'unknown'),
            evidence_lines         = evidence_lines,
            migration_indicated    = diagnosis.get('migration_indicated', False),
            symbol_requirement     = diagnosis.get('symbol_requirement',  'none'),
            migration_urgency      = migration_urgency,
            shapes_block           = shapes_block,
            current_url            = current_url,
            page_title             = page_title,
            links                  = links,
            buttons                = buttons,
            inputs                 = inputs,
            boundary_pressure      = boundary_pressure,
            token_budget_remaining = token_budget_remaining,
            n_shapes               = len(shapes),
        )

        response, tokens_used = self.llm.call(prompt)
        candidates = self._parse_trajectories(response)

        return TrajectoryManifold(
            candidates          = candidates,
            enumeration_context = {
                'tokens_used':  tokens_used,
                'source':       'SYMBOL_GROUNDER_V16',
                'binding_dim':  diagnosis.get('binding_dim'),
                'cause_class':  diagnosis.get('cause_class'),
                'migration':    diagnosis.get('migration_indicated', False),
                **context,
            },
        )

    def ground_symbol(self,
                      action_type:  str,
                      spec:         GroundingSpec,
                      affordances:  Dict,
                      ) -> Optional[str]:
        """
        v16.1: Fill a single symbolic blank for an action the field geometry selected.

        The Triad has already decided:
          - WHAT type of action to take (action_type)
          - WHY: which channels need signal (spec.dark_channels / desired_delta)

        This method returns the concrete token that action needs:
          navigate → URL string
          fill     → text string
          type     → text string
          evaluate → JavaScript expression string
          python   → Python code string

        Returns None on any failure. Caller falls back to observe.

        Each prompt is < 100 tokens in. Output is one token or short expression.
        No reasoning requested. No explanation permitted.
        Not called from the micro-perturbation loop (grounding_spec=None gates that).
        """
        self.call_count += 1
        links_json = json.dumps(affordances.get('links', [])[:10])

        try:
            if action_type == 'navigate':
                prompt = NAVIGATE_GROUNDING_PROMPT.format(
                    current_url     = spec.current_url,
                    page_title      = spec.page_title,
                    sensing_targets = spec.sensing_target_summary(),
                    links           = links_json,
                )
                response, _ = self.llm.call(prompt)
                url = response.strip().split()[0] if response.strip() else ''
                if url.startswith('http'):
                    return url
                return None

            elif action_type in ('fill', 'type'):
                inputs = affordances.get('inputs', [])
                if inputs:
                    inp         = inputs[0]
                    input_type  = inp.get('type', 'text')
                    placeholder = inp.get('placeholder', '')
                else:
                    input_type  = 'text'
                    placeholder = ''
                prompt = FILL_GROUNDING_PROMPT.format(
                    current_url     = spec.current_url,
                    input_type      = input_type,
                    placeholder     = placeholder,
                    sensing_targets = spec.sensing_target_summary(),
                )
                response, _ = self.llm.call(prompt)
                text = response.strip()
                if text and len(text) < 500:
                    return text
                return None

            elif action_type == 'evaluate':
                prompt = EVALUATE_GROUNDING_PROMPT.format(
                    current_url   = spec.current_url,
                    dark_channels = ", ".join(spec.dark_channels[:5]) or "none",
                    desired_delta = spec.desired_delta_summary(),
                )
                response, _ = self.llm.call(prompt)
                code = response.strip()
                if code.startswith('```'):
                    code = '\n'.join(
                        l for l in code.split('\n') if not l.startswith('```')
                    ).strip()
                if code:
                    return code
                return None

            elif action_type == 'python':
                import os
                prompt = PYTHON_GROUNDING_PROMPT.format(
                    cwd             = os.getcwd(),
                    desired_delta   = spec.desired_delta_summary(),
                    sensing_targets = spec.sensing_target_summary(),
                )
                response, _ = self.llm.call(prompt)
                code = response.strip()
                if code.startswith('```'):
                    code = '\n'.join(
                        l for l in code.split('\n') if not l.startswith('```')
                    ).strip()
                if code:
                    return code
                return None

            else:
                return None

        except Exception:
            return None

    def _parse_trajectories(self, response: str) -> List[TrajectoryCandidate]:
        """Parse LLM response with progressive degradation. Unchanged from v15."""
        try:
            cleaned = self._extract_json_block(response)
            if cleaned:
                return self._validate_and_convert(json.loads(cleaned))
            repaired = self._attempt_json_repair(response)
            if repaired:
                return self._validate_and_convert(json.loads(repaired))
            partial = self._extract_partial_trajectories(response)
            if partial:
                return partial
            return self._generate_fallback_trajectory()
        except Exception:
            return self._generate_fallback_trajectory()

    def _extract_json_block(self, response: str) -> Optional[str]:
        if "```json" in response:
            parts = response.split("```json")
            if len(parts) > 1:
                return parts[1].split("```")[0].strip()
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                return parts[1].strip()
        response = response.strip()
        start = response.find('[')
        end   = response.rfind(']') + 1
        if start >= 0 and end > start:
            return response[start:end]
        return None

    def _attempt_json_repair(self, response: str) -> Optional[str]:
        cleaned = response
        for marker in ["Here are", "Here's", "I've enumerated", "The trajectories"]:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[-1]
        cleaned = cleaned.replace('}\n{', '},\n{').replace('} {', '}, {').strip()
        if cleaned.startswith('{') and not cleaned.startswith('['):
            cleaned = f'[{cleaned}]'
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            return None

    def _extract_partial_trajectories(self, response: str) -> List[TrajectoryCandidate]:
        candidates = []
        for match in re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response):
            try:
                obj = json.loads(match)
                if 'steps' in obj and isinstance(obj['steps'], list):
                    candidate = TrajectoryCandidate(
                        steps                            = obj.get('steps', []),
                        rationale                        = obj.get('rationale', 'Partially recovered'),
                        estimated_coherence_preservation = float(obj.get('estimated_coherence_preservation', 0.5)),
                        estimated_optionality_delta      = float(obj.get('estimated_optionality_delta', 0.0)),
                        reversibility_point              = int(obj.get('reversibility_point', 0)),
                    )
                    if len(candidate.steps) > 0:
                        candidates.append(candidate)
            except Exception:
                continue
        return candidates

    def _generate_fallback_trajectory(self) -> List[TrajectoryCandidate]:
        return [TrajectoryCandidate(
            steps                            = [{'action_type': 'observe', 'parameters': {}}],
            rationale                        = 'Symbol grounding fallback — observation only',
            estimated_coherence_preservation = 0.3,
            estimated_optionality_delta      = 0.0,
            reversibility_point              = 0,
        )]

    def _validate_and_convert(self,
                               trajectory_dicts) -> List[TrajectoryCandidate]:
        if not isinstance(trajectory_dicts, list):
            return []
        candidates = []
        for traj_dict in trajectory_dicts:
            try:
                candidate = TrajectoryCandidate(
                    steps                            = traj_dict.get('steps', []),
                    rationale                        = traj_dict.get('rationale', 'No rationale'),
                    estimated_coherence_preservation = float(traj_dict.get('estimated_coherence_preservation', 0.5)),
                    estimated_optionality_delta      = float(traj_dict.get('estimated_optionality_delta', 0.0)),
                    reversibility_point              = int(traj_dict.get('reversibility_point', 0)),
                )
                if 0 < len(candidate.steps) <= 50:
                    if all(isinstance(s, dict) and 'action_type' in s
                           for s in candidate.steps):
                        candidates.append(candidate)
            except (KeyError, ValueError, TypeError):
                continue
        return candidates


