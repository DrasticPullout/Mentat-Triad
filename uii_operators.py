"""
uii_operators.py — v16
DASS operator implementations: SensingOperator, CompressionOperator,
PredictionOperator, CoherenceOperator, SelfModifyingOperator.

Each exposes to_scalar_proxy() for backward compat with existing Φ field.
SelfModifyingOperator replaces the old scalar SMO — plasticity/rigidity
logic preserved exactly; now governs per-layer update rates.

v16 additions (targeted — all existing logic preserved exactly):
  PredictionOperator.covariance_matrix()         — Σ_P as n×n matrix
  PredictionOperator.simulate_covariance_update()— hypothetical Σ_P for action
  CoherenceOperator.current_configuration_vector()  — coverage vec for H_K
  CoherenceOperator.peak_configuration_vector()     — peak coverage vec for H_K
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — SensingOperator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SensingChannel:
    channel_id:  str
    active:      bool
    signal_rate: float   # events/observations per step — real measurement
    last_delta:  float   # magnitude of last observed change
    coverage:    float   # fraction of this channel's domain currently reachable


# Active by default: browser, process_self, os_signals, resource_cpu,
# resource_memory, clock, clock_rate, env_vars, api_llm, entropy_source
DEFAULT_CHANNELS: Dict[str, SensingChannel] = {
    # Browser
    'browser':           SensingChannel('browser',           True,  0.0, 0.0, 0.0),
    # Filesystem
    'filesystem_read':   SensingChannel('filesystem_read',   False, 0.0, 0.0, 0.0),
    'filesystem_watch':  SensingChannel('filesystem_watch',  False, 0.0, 0.0, 0.0),
    'filesystem_io':     SensingChannel('filesystem_io',     False, 0.0, 0.0, 0.0),
    # Network
    'network_http':      SensingChannel('network_http',      False, 0.0, 0.0, 0.0),
    'network_socket':    SensingChannel('network_socket',    False, 0.0, 0.0, 0.0),
    'network_dns':       SensingChannel('network_dns',       False, 0.0, 0.0, 0.0),
    'network_websocket': SensingChannel('network_websocket', False, 0.0, 0.0, 0.0),
    'network_bandwidth': SensingChannel('network_bandwidth', False, 0.0, 0.0, 0.0),
    # Processes
    'process_self':      SensingChannel('process_self',      True,  0.0, 0.0, 1.0),
    'process_children':  SensingChannel('process_children',  False, 0.0, 0.0, 0.0),
    'process_system':    SensingChannel('process_system',    False, 0.0, 0.0, 0.0),
    'os_signals':        SensingChannel('os_signals',        True,  0.0, 0.0, 1.0),
    # IPC
    'ipc_pipe':          SensingChannel('ipc_pipe',          False, 0.0, 0.0, 0.0),
    'ipc_queue':         SensingChannel('ipc_queue',         False, 0.0, 0.0, 0.0),
    'ipc_sharedmem':     SensingChannel('ipc_sharedmem',     False, 0.0, 0.0, 0.0),
    # System resources
    'resource_cpu':      SensingChannel('resource_cpu',      True,  1.0, 0.0, 1.0),
    'resource_memory':   SensingChannel('resource_memory',   True,  1.0, 0.0, 1.0),
    'resource_gpu':      SensingChannel('resource_gpu',      False, 0.0, 0.0, 0.0),
    'resource_disk':     SensingChannel('resource_disk',     False, 0.0, 0.0, 0.0),
    # Time
    'clock':             SensingChannel('clock',             True,  1.0, 0.0, 1.0),
    'clock_rate':        SensingChannel('clock_rate',        True,  1.0, 0.0, 1.0),
    'env_vars':          SensingChannel('env_vars',          True,  0.0, 0.0, 1.0),
    'system_logs':       SensingChannel('system_logs',       False, 0.0, 0.0, 0.0),
    # External services
    'api_llm':           SensingChannel('api_llm',           True,  0.0, 0.0, 1.0),
    'api_external':      SensingChannel('api_external',      False, 0.0, 0.0, 0.0),
    'database':          SensingChannel('database',          False, 0.0, 0.0, 0.0),
    'email':             SensingChannel('email',             False, 0.0, 0.0, 0.0),
    # ssh_remote: migration handshake channel.
    # When a migrated instance starts up it signals back here.
    # ssh_remote becoming active after a migrate action = handshake_received.
    'ssh_remote':        SensingChannel('ssh_remote',        False, 0.0, 0.0, 0.0),
    # Hardware I/O
    'serial_port':       SensingChannel('serial_port',       False, 0.0, 0.0, 0.0),
    'audio_in':          SensingChannel('audio_in',          False, 0.0, 0.0, 0.0),
    'audio_out':         SensingChannel('audio_out',         False, 0.0, 0.0, 0.0),
    'display':           SensingChannel('display',           False, 0.0, 0.0, 0.0),
    'input_devices':     SensingChannel('input_devices',     False, 0.0, 0.0, 0.0),
    # Entropy / human interface
    'entropy_source':    SensingChannel('entropy_source',    True,  0.0, 0.0, 1.0),
    'clipboard':         SensingChannel('clipboard',         False, 0.0, 0.0, 0.0),
    # stdin — human-in-the-loop signal channel
    'stdin':             SensingChannel('stdin',             False, 0.0, 0.0, 0.0),
    # Process internals
    'process_threads':   SensingChannel('process_threads',   False, 0.0, 0.0, 0.0),
    # Extended resources
    'resource_swap':     SensingChannel('resource_swap',     False, 0.0, 0.0, 0.0),
    'resource_fd':       SensingChannel('resource_fd',       False, 0.0, 0.0, 0.0),
    'resource_thermal':  SensingChannel('resource_thermal',  False, 0.0, 0.0, 0.0),
    'resource_battery':  SensingChannel('resource_battery',  False, 0.0, 0.0, 0.0),
    # Python runtime health
    'gc_pressure':       SensingChannel('gc_pressure',       False, 0.0, 0.0, 0.0),
    # Ledger proximity channel — the Triad sensing its own basin geometry
    # How close is current state to the best basin ever recorded in the ledger.
    # Populated in step() from peak Vol_opt comparison — not a system probe.
    'ledger_proximity': SensingChannel('ledger_proximity', True, 0.0, 0.0, 0.0),
}


class SensingOperator:
    """
    S: {env} → {S_i}   Channels over which Reality is sensed.

    Immutable update: apply() returns a new SensingOperator.
    Dark channels decay coverage by 0.9 per step — never removed from domain.
    """

    def __init__(self, channels: Dict[str, SensingChannel]):
        self.channels = channels

    def apply(self, env_signal: Dict, internal_state: 'CompressionOperator') -> 'SensingOperator':
        """
        Update sensing channels from env_signal.
        Dark (inactive) channels decay coverage.
        Returns new SensingOperator — does not mutate.

        S: E × I → S'  (DASS invariant)
        i_gate implements the I term: compression quality governs how much new
        environmental signal is admitted into coverage. At I=0 → 50% admitted;
        at I=1 → 100% admitted. Gate lives here — not in Reality — because
        Reality is a substrate-agnostic measurement source that must not read
        internal operator state.
        """
        # I-gate: compression quality determines signal admission rate.
        # internal_state is CompressionOperator; scalar_proxy ∈ [0, 1].
        i_quality = internal_state.to_scalar_proxy() if internal_state is not None else 0.5
        i_gate    = 0.5 + 0.5 * i_quality   # [0.5, 1.0]

        updated = {}
        for cid, channel in self.channels.items():
            if not channel.active:
                sig = env_signal.get(cid)
                incoming_coverage = (
                    float(sig.get('coverage', 0.0)) if isinstance(sig, dict) else 0.0
                )
                if incoming_coverage > 0.0:
                    # Dark channel receiving real signal for the first time — activate.
                    # Coverage initialises from the probe; the geometry takes it from here.
                    updated[cid] = dataclasses.replace(
                        channel,
                        active      = True,
                        coverage    = incoming_coverage,
                        signal_rate = float(sig.get('rate',      0.0)),
                        last_delta  = float(sig.get('magnitude', 0.0)),
                    )
                else:
                    # Truly dark — decay and stay dark.
                    updated[cid] = dataclasses.replace(channel, coverage=channel.coverage * 0.9)
                continue

            sig = env_signal.get(cid)
            if sig is None:
                updated[cid] = channel
                continue

            if isinstance(sig, dict):
                magnitude        = float(sig.get('magnitude', 0.0))
                rate             = float(sig.get('rate',      channel.signal_rate))
                incoming_cov     = float(sig.get('coverage',  channel.coverage))
            else:
                magnitude        = float(sig)
                rate             = channel.signal_rate
                incoming_cov     = channel.coverage

            # EMA update for signal_rate
            alpha    = 0.1
            new_rate = (1 - alpha) * channel.signal_rate + alpha * rate
            new_rate = float(np.clip(new_rate, 0.0, 10.0))

            # Coverage update gated by I: S: E × I → S'
            # The gate blends new coverage toward current at low I, admitting
            # fully at high I. This is not punitive — it reflects that a poorly
            # compressed system cannot reliably incorporate novel sensing surface.
            gated_coverage = channel.coverage + i_gate * (incoming_cov - channel.coverage)
            new_coverage   = float(np.clip(gated_coverage, 0.0, 1.0))

            updated[cid] = dataclasses.replace(
                channel,
                signal_rate = new_rate,
                last_delta  = magnitude,
                coverage    = new_coverage,
            )

        return SensingOperator(channels=updated)

    def to_scalar_proxy(self) -> float:
        active = [c for c in self.channels.values() if c.active]
        if not active:
            return 0.0
        return float(np.mean([c.coverage * c.signal_rate for c in active]))

    def domain_size(self) -> int:
        return sum(1 for c in self.channels.values() if c.active)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — CompressionOperator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalEdge:
    source:      str
    target:      str
    weight:      float   # EMA co-movement
    confidence:  float   # min(1.0, observations / 200)
    lag:         int     # 0 = synchronous; lag detection future L3 axis candidate


class CompressionOperator:
    """
    I: {S_i} → C   Causal graph over channels.

    causal_graph keys are (source_id, target_id) tuples.
    Immutable update: apply() returns a new CompressionOperator.
    """

    EMA_ALPHA = 0.05   # same as existing coupling matrix

    def __init__(self,
                 causal_graph:      Dict[Tuple[str, str], CausalEdge],
                 residual_variance: Dict[str, float],
                 prediction_errors: deque,
                 observation_count: int):
        self.causal_graph      = causal_graph
        self.residual_variance = residual_variance
        self.prediction_errors = prediction_errors
        self.observation_count = observation_count

    def apply(self, sensing_history: deque) -> 'CompressionOperator':
        """
        Update causal graph from recent sensing history.
        Only active channels generate co-movement edges.
        Returns new CompressionOperator — does not mutate.
        """
        if len(sensing_history) < 2:
            return CompressionOperator(
                causal_graph      = dict(self.causal_graph),
                residual_variance = dict(self.residual_variance),
                prediction_errors = deque(self.prediction_errors, maxlen=self.prediction_errors.maxlen),
                observation_count = self.observation_count,
            )

        latest = sensing_history[-1]
        prev   = sensing_history[-2]

        # Active channels in both steps
        active_cids = [
            cid for cid, ch in latest.channels.items()
            if ch.active and prev.channels.get(cid, SensingChannel(cid, False, 0, 0, 0)).active
        ]

        new_graph = dict(self.causal_graph)
        new_obs   = self.observation_count + 1

        for src in active_cids:
            for tgt in active_cids:
                if src == tgt:
                    continue
                src_delta = latest.channels[src].last_delta
                tgt_delta = latest.channels[tgt].last_delta

                if abs(src_delta) < 1e-8:
                    continue

                co_movement = float(np.clip(tgt_delta / (src_delta + 1e-8), -2.0, 2.0))

                key = (src, tgt)
                if key in new_graph:
                    existing = new_graph[key]
                    new_weight = (1 - self.EMA_ALPHA) * existing.weight + self.EMA_ALPHA * co_movement
                    new_graph[key] = dataclasses.replace(
                        existing,
                        weight     = float(np.clip(new_weight, -2.0, 2.0)),
                        confidence = existing.confidence,  # confidence earned in SMO after full loop
                    )
                else:
                    new_graph[key] = CausalEdge(
                        source     = src,
                        target     = tgt,
                        weight     = co_movement,
                        confidence = 0.0,   # no evidence yet — SMO will earn it
                        lag        = 0,
                    )

        # Residual variance: variance of last_delta per channel
        new_residual = dict(self.residual_variance)
        for cid in active_cids:
            deltas = [
                s.channels[cid].last_delta
                for s in sensing_history
                if s.channels.get(cid) and s.channels[cid].active
            ]
            if len(deltas) >= 2:
                new_residual[cid] = float(np.var(deltas))

        return CompressionOperator(
            causal_graph      = new_graph,
            residual_variance = new_residual,
            prediction_errors = deque(self.prediction_errors, maxlen=self.prediction_errors.maxlen),
            observation_count = new_obs,
        )

    def to_scalar_proxy(self) -> float:
        if not self.causal_graph:
            return 0.0
        mean_confidence = float(np.mean([e.confidence for e in self.causal_graph.values()]))
        mean_residual   = float(np.mean(list(self.residual_variance.values()))) \
                          if self.residual_variance else 1.0
        return float(np.clip(mean_confidence * (1.0 - mean_residual), 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — PredictionOperator
# ──────────────────────────────────────────────────────────────────────────────

NOISE_FLOOR = 0.05   # threshold at which prediction error degrades horizon


@dataclass
class ChannelPrediction:
    channel_id:      str
    predicted_delta: float
    confidence:      float
    horizon:         int               # steps forward before noise floor
    error_history:   deque             # maxlen=20


class PredictionOperator:
    """
    P: C × M → {τᵢ}   Realized prediction horizon + virtual trajectories.

    Immutable update: apply() and observe_outcome() return new instances.
    """

    def __init__(self,
                 channel_predictions:  Dict[str, ChannelPrediction],
                 realized_horizon:     int,
                 prediction_accuracy:  Dict[str, float]):
        self.channel_predictions  = channel_predictions
        self.realized_horizon     = realized_horizon
        self.prediction_accuracy  = prediction_accuracy

    def apply(self, compression: CompressionOperator) -> 'PredictionOperator':
        """
        Generate per-channel predictions from causal graph.
        Aggregates incoming edge weights for each channel.
        Returns new PredictionOperator — does not mutate.
        """
        new_predictions: Dict[str, ChannelPrediction] = {}

        # Collect all channels that appear as targets in the causal graph
        target_channels: Dict[str, List[CausalEdge]] = {}
        for (src, tgt), edge in compression.causal_graph.items():
            if tgt not in target_channels:
                target_channels[tgt] = []
            target_channels[tgt].append(edge)

        for cid, edges in target_channels.items():
            # Aggregate incoming edge weights weighted by confidence
            total_weight = sum(e.weight * e.confidence for e in edges)
            total_conf   = sum(e.confidence for e in edges)
            predicted    = total_weight / total_conf if total_conf > 1e-6 else 0.0
            confidence   = min(1.0, total_conf / len(edges)) if edges else 0.0

            prior = self.channel_predictions.get(cid)
            if prior is not None:
                err_hist = deque(prior.error_history, maxlen=20)
                horizon  = prior.horizon
            else:
                err_hist = deque(maxlen=20)
                horizon  = 0

            new_predictions[cid] = ChannelPrediction(
                channel_id      = cid,
                predicted_delta = float(predicted),
                confidence      = float(confidence),
                horizon         = horizon,
                error_history   = err_hist,
            )

        return PredictionOperator(
            channel_predictions  = new_predictions,
            realized_horizon     = self.realized_horizon,
            prediction_accuracy  = dict(self.prediction_accuracy),
        )

    def observe_outcome(self,
                        sensing:     SensingOperator,
                        compression: CompressionOperator
                        ) -> Tuple['PredictionOperator', Dict[str, float]]:
        """
        Compare predictions against actual SensingOperator.
        Update per-channel horizon: +1 when error < NOISE_FLOOR, -1 when above.
        Returns (updated_PredictionOperator, error_dict).
        error_dict is used by SMO (plasticity adaptation) and CoherenceOperator (p_a).
        P errors do not feed back to CompressionOperator — I creates C from S;
        whether P can predict from C is P's problem, not evidence that C is wrong.
        """
        error_dict:   Dict[str, float] = {}
        new_preds     = dict(self.channel_predictions)
        new_accuracy  = dict(self.prediction_accuracy)
        new_horizon   = self.realized_horizon

        for cid, pred in self.channel_predictions.items():
            ch = sensing.channels.get(cid)
            if ch is None or not ch.active:
                continue

            actual = ch.last_delta
            error  = abs(actual - pred.predicted_delta)
            error_dict[cid] = error

            # Update error history
            err_hist = deque(pred.error_history, maxlen=20)
            err_hist.append(error)

            # Per-channel horizon update
            new_ch_horizon = pred.horizon
            if error < NOISE_FLOOR:
                new_ch_horizon = min(50, new_ch_horizon + 1)
            else:
                new_ch_horizon = max(0, new_ch_horizon - 1)

            new_accuracy[cid] = (
                0.9 * new_accuracy.get(cid, 0.5) + 0.1 * (1.0 - min(1.0, error))
            )

            new_preds[cid] = dataclasses.replace(
                pred,
                error_history = err_hist,
                horizon       = new_ch_horizon,
            )

        # Global realized_horizon: mean of per-channel horizons
        if new_preds:
            new_horizon = int(round(np.mean([p.horizon for p in new_preds.values()])))
            new_horizon = int(np.clip(new_horizon, 0, 50))

        updated = PredictionOperator(
            channel_predictions  = new_preds,
            realized_horizon     = new_horizon,
            prediction_accuracy  = new_accuracy,
        )
        return updated, error_dict

    def test_virtual(self,
                     compression:        'CompressionOperator',
                     candidate_action:   str,
                     phi_field,
                     sensing:            'SensingOperator',
                     coupling_estimator  = None) -> Dict[str, float]:
        """
        Project action forward through channel space and causal graph.
        Returns a channel-keyed delta dict suitable for ⟨delta, ∇Φ⟩ scoring.
        Nothing written to actual state — reversibility is structural.

        Two stages:
          1. Primary: direct channel influence from the PRIMARY prior table.
          2. Secondary: propagate primary deltas through the causal graph.

        The empirical coupling_estimator branch (v16.2.1) has been removed.
        That branch read SIPA-space affordance deltas and projected them back
        through an inverse CHANNEL_TO_DIM mapping — reversing the causal
        direction (channels produce SIPA; SIPA cannot reconstruct channels).
        Empirical grounding requires Reality to emit per-channel deltas
        directly; that is a future addition. PRIMARY is the only source
        until then. coupling_estimator parameter retained for call-site compat.
        """
        PRIMARY: Dict[str, Dict[str, float]] = {
            'navigate':    {'browser': 0.1, 'network_http': 0.05, 'network_dns': 0.02},
            'click':       {'browser': 0.05},
            'read':        {'browser': 0.03},
            'scroll':      {'browser': 0.02},
            'fill':        {'browser': 0.03},
            'type':        {'browser': 0.03},
            'evaluate':    {'browser': 0.02, 'process_self': 0.01},
            'observe':     {'browser': 0.01},
            'delay':       {},
            'python':      {'process_self': 0.05, 'resource_cpu': 0.1,
                            'resource_memory': 0.05, 'filesystem_io': 0.03},
            'migrate':     {'ssh_remote': 0.3, 'process_self': 0.1,
                            'network_socket': 0.05},
            'llm_query':   {'api_llm': 0.1, 'network_http': 0.03},
            'query_agent': {},
        }

        primary: Dict[str, float] = dict(PRIMARY.get(candidate_action, {}))

        delta: Dict[str, float] = dict(primary)
        for src, magnitude in primary.items():
            for (edge_src, edge_tgt), edge in compression.causal_graph.items():
                if edge_src == src:
                    secondary = magnitude * edge.weight * edge.confidence
                    delta[edge_tgt] = delta.get(edge_tgt, 0.0) + secondary

        return delta

    def to_scalar_proxy(self) -> float:
        return float(np.clip(self.realized_horizon / 50.0, 0.0, 1.0))

    def to_grounded_proxy(self,
                           sensing:     SensingOperator,
                           compression: CompressionOperator) -> float:
        """
        P: C → C' quality. Measures forward model accuracy and horizon.

        From the invariant: ∃ operator P: C → C' such that predicted ΔC
        remains bounded and reversible.

        Two phases based on whether accuracy evidence exists:

        Bootstrap (no accuracy evidence yet):
          Returns horizon_norm alone. At step 1, realized_horizon = 50 (max),
          so P = 1.0 — all futures are open, nothing has been ruled out.
          This correctly represents maximum optionality, not overconfidence:
          accuracy is 0 (no predictions verified), but horizon is maximal
          (no futures have been eliminated by error either).
          As observe_outcome() accumulates errors, per-channel horizons
          decline where predictions fail, and the formula transitions to
          the evidence-weighted phase.

        Evidence phase (accuracy observations present):
          P = mean_accuracy × (0.5 + 0.5 × horizon_norm)
          Both factors required for high P:
            accurate over short horizon  = moderate P (can predict one step)
            accurate over long horizon   = high P     (bounded multi-step)
          This matches the invariant: bounded AND reversible.

        CRK note:
          C3 (non-internalization) fires on failure outcomes degrading A,
          not on low bootstrap accuracy — early P is not a failure state.
          C4 (reality) fires on false confidence, not on high horizon at
          bootstrap — open optionality is epistemically correct.
        """
        active_acc = [
            acc for cid, acc in self.prediction_accuracy.items()
            if sensing.channels.get(cid) and sensing.channels[cid].active
        ]

        horizon_norm = float(np.clip(self.realized_horizon / 50.0, 0.0, 1.0))

        if not active_acc:
            # Bootstrap: no accuracy evidence yet.
            # P's value to A is purely optionality — all futures are open.
            return horizon_norm

        mean_accuracy = float(np.mean(active_acc))
        return float(np.clip(mean_accuracy * (0.5 + 0.5 * horizon_norm), 0.0, 1.0))

    # ── v16 additions ─────────────────────────────────────────────────────────

    def covariance_matrix(self,
                          sensing:     'SensingOperator',
                          compression: 'CompressionOperator',
                          ) -> Tuple[np.ndarray, List[str]]:
        """
        Build Σ_P over active channels in channel space.

        Returns (sigma_p, active_channel_list).
        sigma_p is n×n, symmetrized.
        active_channel_list defines the axis ordering (index ↔ channel_id).

        Construction (same as to_grounded_proxy, returns matrix not scalar):
          Diagonal:     sigma_p[i,i] = coverage_i
          Off-diagonal: sigma_p[i,j] += weight(i→j) * conf(i→j) * coverage_i

        Symmetrize: (sigma_p + sigma_p.T) / 2

        Returns (zeros((0,0)), []) if no active channels.

        Called by PhiField.compute_hessian() (H_O construction) and by
        expected_optionality_gain() in uii_geometry.py. Not recomputed
        inside scoring loops — caller caches the result for the full step.
        """
        channels = sensing.channels
        active = [cid for cid, ch in channels.items() if ch.active]
        n = len(active)
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

    def simulate_covariance_update(self,
                                    action:      str,
                                    compression: 'CompressionOperator',
                                    sensing:     'SensingOperator',
                                    ) -> Tuple[np.ndarray, List[str]]:
        """
        Hypothetical Σ_P if action were executed — does not mutate state.

        Process:
          1. Call test_virtual(compression, action, phi_field=None, sensing)
             to get channel delta dict. phi_field=None is safe — test_virtual
             accepts but never dereferences phi_field in its body.
          2. Build hypothetical coverage dict: apply deltas to current
             coverage values, clipped to [0.0, 1.0].
          3. Reconstruct Σ_P using hypothetical coverages and real causal graph.
          4. Return (sigma_p_hypothetical, active_channel_list).

        At bootstrap (empty causal graph): secondary propagation is zero;
        only PRIMARY lookup deltas apply. The approximation degrades
        gracefully as the causal graph fills in — correct behavior.

        Returns same shape as covariance_matrix(): (n×n matrix, active_list).
        Returns (zeros((0,0)), []) if no active channels.

        Used by expected_optionality_gain() in uii_geometry.py to compute
        E[Δlog(O(a))] pre-action, once per step, shared between viable set
        construction and score_actions(). Not recomputed inside either.
        """
        delta = self.test_virtual(compression, action, phi_field=None, sensing=sensing)

        channels = sensing.channels
        active = [cid for cid, ch in channels.items() if ch.active]
        n = len(active)
        if n == 0:
            return np.zeros((0, 0)), []

        idx = {cid: i for i, cid in enumerate(active)}

        # Apply coverage deltas hypothetically — never touches real state
        hyp_coverage = {
            cid: float(np.clip(channels[cid].coverage + delta.get(cid, 0.0), 0.0, 1.0))
            for cid in active
        }

        sigma_p = np.zeros((n, n))

        for i, cid in enumerate(active):
            sigma_p[i, i] = hyp_coverage[cid]

        for (src, tgt), edge in compression.causal_graph.items():
            if src in idx and tgt in idx:
                sigma_p[idx[src], idx[tgt]] += (
                    edge.weight * edge.confidence * hyp_coverage[src]
                )

        sigma_p = (sigma_p + sigma_p.T) / 2.0
        return sigma_p, active


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — CoherenceOperator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OperatorConsistencyCheck:
    """v15.3: full S→I→P→A→SMO→S loop. Four operators, four transitions."""
    s_i_consistency: float   # active channels covered by causal graph
    i_p_consistency: float   # confident edges covered by predictions
    p_a_consistency: float   # P grounded enough to inform A  [v15.3]
    smo_consistency: float   # SMO invariants satisfied (reversible)  [v15.3]
    loop_closure:    float   # geometric mean of all four


class CoherenceOperator:
    """
    A: measures internal consistency of the S→I→P→A→SMO→S loop.

    v15.3: full four-transition loop closure.
    Only operator that sees the full system.
    Immutable update: apply() returns new CoherenceOperator.
    """

    def __init__(self,
                 consistency:         OperatorConsistencyCheck,
                 consistency_history: deque,
                 loop_signature:      Dict[str, float],
                 signature_deviation: float,
                 self_model:          Dict):
        self.consistency         = consistency
        self.consistency_history = consistency_history
        self.loop_signature      = loop_signature
        self.signature_deviation = signature_deviation
        self.self_model          = self_model

    def apply(self,
              sensing:      SensingOperator,
              compression:  CompressionOperator,
              prediction:   PredictionOperator,
              smo_updates:  Optional[List] = None) -> 'CoherenceOperator':
        """
        Compute full four-transition loop closure: S→I→P→A→SMO→S.

        smo_updates: List[SMOUpdate] from SelfModifyingOperator.apply() this step.
                     None → smo_consistency = 1.0 (bootstrap fallback).
        Returns new CoherenceOperator — does not mutate.
        """
        # ── s_i: S→I ─────────────────────────────────────────────────────────
        # Fraction of active sensing channels that appear in the causal graph.
        # Bootstrap: empty graph means nothing is inconsistent yet — not that
        # everything has failed. s_i = 1.0 until graph has edges to measure against.
        active_cids = {cid for cid, ch in sensing.channels.items() if ch.active}
        if not compression.causal_graph:
            s_i = 1.0   # bootstrap: no graph → no inconsistency
        else:
            graph_cids  = set()
            for (src, tgt) in compression.causal_graph:
                graph_cids.add(src)
                graph_cids.add(tgt)
            s_i = len(active_cids & graph_cids) / len(active_cids) if active_cids else 0.0

        # ── i_p: I→P — unchanged ─────────────────────────────────────────────
        confident_edges = {
            (src, tgt) for (src, tgt), edge in compression.causal_graph.items()
            if edge.confidence > 0.3
        }
        predicted_cids = set(prediction.channel_predictions.keys())
        if confident_edges:
            covered = sum(
                1 for (src, tgt) in confident_edges
                if src in predicted_cids or tgt in predicted_cids
            )
            i_p = covered / len(confident_edges)
        else:
            i_p = 1.0   # no confident edges to be inconsistent with

        # ── p_a: P→A — NEW v15.3 ─────────────────────────────────────────────
        # Is P grounded enough to meaningfully inform A?
        # P informs A only when grounded by S and I.
        # p_a = P.to_grounded_proxy() directly.
        p_a = float(np.clip(
            prediction.to_grounded_proxy(sensing, compression), 0.0, 1.0
        ))

        # ── smo_consistency: A→SMO→S — NEW v15.3 ─────────────────────────────
        # SMO reports via SMOUpdate.reversible.
        # reversible=True means bounded + reversible invariants satisfied.
        # Withheld updates excluded — withheld = A correctly gated SMO = loop working.
        # smo_consistency = fraction of non-withheld updates where reversible=True.
        if smo_updates is not None:
            ran = [u for u in smo_updates if not u.withheld]
            if ran:
                smo_consistency = float(
                    sum(1 for u in ran if u.reversible) / len(ran)
                )
            else:
                # All updates withheld — A correctly gated SMO. Loop intact.
                smo_consistency = 1.0
        else:
            smo_consistency = 1.0   # bootstrap: no SMO data yet, don't penalize

        # ── loop_closure: geometric mean of all four ──────────────────────────
        loop_closure = float(np.clip(
            (s_i * i_p * p_a * smo_consistency) ** (1.0 / 4.0), 0.0, 1.0
        ))

        consistency = OperatorConsistencyCheck(
            s_i_consistency  = float(np.clip(s_i,             0.0, 1.0)),
            i_p_consistency  = float(np.clip(i_p,             0.0, 1.0)),
            p_a_consistency  = float(np.clip(p_a,             0.0, 1.0)),
            smo_consistency  = float(np.clip(smo_consistency, 0.0, 1.0)),
            loop_closure     = loop_closure,
        )

        new_history = deque(self.consistency_history, maxlen=self.consistency_history.maxlen)
        new_history.append(consistency.loop_closure)

        # Update loop signature — only learn from coherent steps (> 0.7)
        new_signature = dict(self.loop_signature)
        if loop_closure > 0.7:
            new_signature = self._update_loop_signature(new_signature, sensing, compression, prediction)

        # Signature deviation
        sig_dev = self._compute_signature_deviation(new_signature, sensing, compression, prediction)

        # Update self-model — only on sufficiently coherent steps (> 0.6)
        new_self_model = dict(self.self_model)
        if loop_closure > 0.6:
            new_self_model = self._update_self_model(new_self_model, sensing, compression, prediction)

        return CoherenceOperator(
            consistency         = consistency,
            consistency_history = new_history,
            loop_signature      = new_signature,
            signature_deviation = sig_dev,
            self_model          = new_self_model,
        )

    def _update_loop_signature(self,
                                signature:   Dict[str, float],
                                sensing:     SensingOperator,
                                compression: CompressionOperator,
                                prediction:  PredictionOperator) -> Dict[str, float]:
        """
        Learn normal loop signature from coherent steps.
        Cold start (first 5 steps): signature empty, deviation treated as 0.
        """
        alpha = 0.05
        new_sig = dict(signature)

        new_sig['active_channels'] = (
            (1 - alpha) * new_sig.get('active_channels', sensing.domain_size())
            + alpha * sensing.domain_size()
        )
        new_sig['graph_edges'] = (
            (1 - alpha) * new_sig.get('graph_edges', len(compression.causal_graph))
            + alpha * len(compression.causal_graph)
        )
        new_sig['predictions'] = (
            (1 - alpha) * new_sig.get('predictions', len(prediction.channel_predictions))
            + alpha * len(prediction.channel_predictions)
        )
        new_sig['mean_confidence'] = (
            (1 - alpha) * new_sig.get('mean_confidence', compression.to_scalar_proxy())
            + alpha * compression.to_scalar_proxy()
        )

        return new_sig

    def _compute_signature_deviation(self,
                                      signature:   Dict[str, float],
                                      sensing:     SensingOperator,
                                      compression: CompressionOperator,
                                      prediction:  PredictionOperator) -> float:
        """
        Deviation of current state from learned loop signature.
        Empty signature → 0.0 deviation (cold start: C1 satisfied).
        """
        if not signature:
            return 0.0

        deviations = []
        if 'active_channels' in signature and signature['active_channels'] > 0:
            deviations.append(abs(sensing.domain_size() - signature['active_channels'])
                              / max(signature['active_channels'], 1))
        if 'graph_edges' in signature and signature['graph_edges'] > 0:
            deviations.append(abs(len(compression.causal_graph) - signature['graph_edges'])
                              / max(signature['graph_edges'], 1))
        if 'predictions' in signature and signature['predictions'] > 0:
            deviations.append(abs(len(prediction.channel_predictions) - signature['predictions'])
                              / max(signature['predictions'], 1))

        return float(np.clip(np.mean(deviations), 0.0, 1.0)) if deviations else 0.0

    def _update_self_model(self,
                            self_model:  Dict,
                            sensing:     SensingOperator,
                            compression: CompressionOperator,
                            prediction:  PredictionOperator) -> Dict:
        """Update self-model from coherent observations."""
        return {
            'active_channel_count': sensing.domain_size(),
            'graph_edge_count':     len(compression.causal_graph),
            'prediction_count':     len(prediction.channel_predictions),
            'realized_horizon':     prediction.realized_horizon,
            'loop_closure':         self.consistency.loop_closure,
        }

    def crk_signal(self) -> Dict[str, float]:
        return {
            'loop_closure':        self.consistency.loop_closure,
            'signature_deviation': self.signature_deviation,
            's_i_consistency':     self.consistency.s_i_consistency,
            'i_p_consistency':     self.consistency.i_p_consistency,
            'p_a_consistency':     self.consistency.p_a_consistency,
            'smo_consistency':     self.consistency.smo_consistency,
        }

    def to_scalar_proxy(self) -> float:
        return float(self.consistency.loop_closure)

    # ── v16 additions ─────────────────────────────────────────────────────────

    def current_configuration_vector(self,
                                      sensing:         'SensingOperator',
                                      active_channels: List[str],
                                      ) -> np.ndarray:
        """
        Current coverage vector in active channel space.

        active_channels: ordered list defining axis correspondence —
            must match the channel_basis ordering from the Hessian call
            so that H_K = -outer(delta, delta) is in the correct space.

        Returns np.ndarray of shape (n,) where n = len(active_channels).
        Channels not present in sensing default to 0.0 coverage.

        Used by PhiField.compute_hessian() to build H_K.
        """
        return np.array([
            sensing.channels[cid].coverage if cid in sensing.channels else 0.0
            for cid in active_channels
        ])

    def peak_configuration_vector(self,
                                   peak_operator_snapshot: Dict,
                                   active_channels:        List[str],
                                   ) -> np.ndarray:
        """
        Peak coverage vector from ledger operator snapshot, in active channel space.

        peak_operator_snapshot: full ledger.operator_snapshot dict.
            Reads ['sensing']['channels'][cid]['coverage'].

        active_channels: ordered list defining axis correspondence —
            same ordering used by current_configuration_vector() so that
            delta = current - peak is a valid vector difference.

        Returns np.ndarray of shape (n,).
        Channels not present in peak snapshot default to 0.0.

        Used by PhiField.compute_hessian() to build H_K:
            delta = current_configuration_vector() - peak_configuration_vector()
            H_K   = -outer(delta, delta)
        """
        peak_channels = peak_operator_snapshot.get('sensing', {}).get('channels', {})
        return np.array([
            float(peak_channels.get(cid, {}).get('coverage', 0.0))
            for cid in active_channels
        ])


# ──────────────────────────────────────────────────────────────────────────────
# SelfModifyingOperator — v15.1 layer-specific SMO
# (Appended per spec — SMO section)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SMOUpdate:
    """
    Proposed update to one operator layer.
    Computed before applying — allows invariant checks before commit.
    """
    layer:           str            # 'S' | 'I' | 'P' | 'A'
    delta_norm:      float          # ‖ΔL‖ — magnitude of proposed change
    reversible:      bool           # can prior state be reconstructed
    withheld:        bool           # invariant check blocked this update
    withheld_reason: Optional[str]


class SelfModifyingOperator:
    """
    SMO: M → M'

    Layer-specific applications:
    SMO_S: S × ΔS → S'   — update sensing channel parameters
    SMO_I: I × ΔI → I'   — update causal graph structure
    SMO_P: P × ΔP → P'   — update prediction parameters
    SMO_A: A × ΔA → A'   — update coherence signature learning rate

    Each layer update is bounded, reversible, and checked against
    attractor preservation and optionality preservation before commit.

    The scalar rigidity mechanism from v14/v15 becomes SMO_S's
    plasticity parameter — it governs how quickly the sensing domain
    responds to new signal. It is not discarded.
    """

    EPSILON = 0.1    # ‖ΔM‖ bound per step
    THETA   = 0.15   # max allowed loop_closure drop after update

    def __init__(self):
        # Scalar plasticity (from original SMO) — now governs SMO_S responsiveness
        self.plasticity:   float = 0.5    # inverse of rigidity; 0=rigid, 1=plastic
        self.rigidity:     float = 0.5    # kept for backward compat log output
        self.recent_error: float = 0.0

        # Per-layer update history — for SMO⁻¹ reconstruction
        self._snapshots: deque = deque(maxlen=10)

    def apply(self,
              sensing:           SensingOperator,
              compression:       CompressionOperator,
              prediction:        PredictionOperator,
              coherence:         CoherenceOperator,
              observed_delta:    Dict[str, float],
              predicted_delta:   Dict[str, float],
              prediction_errors: Dict[str, float],
              ) -> Tuple[SensingOperator, CompressionOperator,
                         PredictionOperator, CoherenceOperator,
                         List[SMOUpdate]]:
        """
        SMO: M → M'

        Applies layer-specific updates in order: S → I → P → A.
        Each layer reads from the updated version of prior layers.
        Each update is bounded and checked before commit.

        Returns updated operators and update log.
        CRK post-action already verified this should run before this is called.
        """
        # Snapshot for SMO⁻¹
        self._snapshots.append({
            'sensing_channels':    {k: dataclasses.asdict(v)
                                    for k, v in sensing.channels.items()},
            'compression_edges':   {str(k): dataclasses.asdict(v)
                                    for k, v in compression.causal_graph.items()},
            'prediction_horizon':  prediction.realized_horizon,
            'coherence_signature': dict(coherence.loop_signature),
        })

        updates = []

        # Update plasticity from recent error — core of old scalar SMO
        mean_error = float(np.mean(list(prediction_errors.values()))) \
                     if prediction_errors else 0.0
        self.recent_error = mean_error

        # Asymmetric update — gain rigidity slowly, lose it fast (preserved from v14)
        if mean_error < 0.02:
            self.plasticity = max(0.1, self.plasticity - 0.01)
        else:
            self.plasticity = min(0.9, self.plasticity + 0.02)
        self.rigidity = 1.0 - self.plasticity

        # SMO_S — update sensing channel parameters
        new_sensing, s_update = self._smo_s(sensing, observed_delta, prediction_errors)
        updates.append(s_update)

        # SMO_I — update causal graph
        new_compression, i_update = self._smo_i(compression, new_sensing, prediction_errors)
        updates.append(i_update)

        # SMO_P — update prediction parameters; check optionality preservation
        new_prediction, p_update = self._smo_p(prediction, new_compression, prediction_errors, coherence)
        updates.append(p_update)

        # SMO_A — update coherence signature learning rate; must not drop loop_closure > THETA
        new_coherence, a_update = self._smo_a(coherence, new_sensing, new_compression, new_prediction)
        updates.append(a_update)

        return new_sensing, new_compression, new_prediction, new_coherence, updates

    def _smo_s(self,
               sensing:           SensingOperator,
               observed_delta:    Dict,
               prediction_errors: Dict) -> Tuple[SensingOperator, SMOUpdate]:
        """
        SMO_S: S × ΔS → S'

        Updates sensing channel parameters based on observed signal.
        Plasticity governs how quickly coverage and signal_rate respond.

        Dark channels that SensingOperator.apply() has already activated this
        step (active=True) are processed normally here. Channels that are still
        dark AND have no incoming signal are passed through unchanged —
        SensingOperator.apply() handles their decay.
        """
        updated_channels = {}
        total_delta      = 0.0

        for cid, channel in sensing.channels.items():
            if not channel.active and observed_delta.get(cid) is None:
                updated_channels[cid] = channel
                continue

            sig = observed_delta.get(cid)
            if isinstance(sig, dict):
                obs_rate     = sig.get('rate',     channel.signal_rate)
                obs_coverage = sig.get('coverage', channel.coverage)
            else:
                obs_rate     = channel.signal_rate
                obs_coverage = channel.coverage

            alpha    = 0.05 * (1.0 + self.plasticity)
            new_rate = (1 - alpha) * channel.signal_rate + alpha * obs_rate
            new_rate = float(np.clip(new_rate, 0.0, 10.0))

            delta_coverage = float(np.clip(obs_coverage - channel.coverage,
                                           -self.EPSILON, self.EPSILON))
            new_coverage   = float(np.clip(channel.coverage + delta_coverage, 0.0, 1.0))

            total_delta += abs(new_rate - channel.signal_rate) + abs(delta_coverage)

            updated_channels[cid] = dataclasses.replace(
                channel,
                signal_rate = new_rate,
                coverage    = new_coverage,
            )

        delta_norm = total_delta / max(len(sensing.channels), 1)

        return (
            SensingOperator(channels=updated_channels),
            SMOUpdate('S', delta_norm, True, False, None)
        )

    def _smo_i(self,
               compression:       CompressionOperator,
               sensing:           SensingOperator,
               prediction_errors: Dict) -> Tuple[CompressionOperator, SMOUpdate]:
        """
        SMO_I: I × ΔI → I'

        Updates causal graph edge weights based on prediction errors.
        Bounded: no single edge weight changes by more than EPSILON per step.
        Reversibility: edge weights change gradually — no edges destroyed.
        """
        updated_graph = {}
        total_delta   = 0.0

        for key, edge in compression.causal_graph.items():
            src, tgt   = key
            src_error  = prediction_errors.get(src, 0.0)
            tgt_error  = prediction_errors.get(tgt, 0.0)
            edge_error = max(src_error, tgt_error)

            alpha = min(0.2, 0.05 * (1.0 + self.plasticity) * (1.0 + edge_error))

            src_ch = sensing.channels.get(src)
            tgt_ch = sensing.channels.get(tgt)
            src_delta = src_ch.last_delta if src_ch else 0.0
            tgt_delta = tgt_ch.last_delta if tgt_ch else 0.0

            if abs(src_delta) > 1e-6:
                co_movement  = float(np.clip(tgt_delta / (src_delta + 1e-8), -2.0, 2.0))
                weight_delta = float(np.clip(
                    alpha * (co_movement - edge.weight),
                    -self.EPSILON, self.EPSILON
                ))
                new_weight = float(np.clip(edge.weight + weight_delta, -2.0, 2.0))

                # Confidence earned from Reality-grounded prediction error.
                # tgt_error comes from P.observe_outcome() — the difference
                # between what P predicted and what S actually measured.
                # The loop must close (S->I->P->A->SMO->S) before confidence
                # is updated; I cannot assess itself. An edge becomes confident
                # only when it has contributed to accurate predictions that
                # Reality has subsequently confirmed.
                prediction_accuracy = 1.0 - min(1.0, tgt_error)
                new_conf = float(np.clip(
                    (1 - alpha) * edge.confidence + alpha * prediction_accuracy,
                    0.0, 1.0
                ))
            else:
                # No signal this cycle — loop completed but reality didn't speak
                # on this edge. Confidence neither earned nor lost.
                new_weight   = edge.weight
                weight_delta = 0.0
                new_conf     = edge.confidence

            total_delta += abs(weight_delta)

            updated_graph[key] = dataclasses.replace(
                edge,
                weight     = new_weight,
                confidence = new_conf,
            )

        delta_norm = total_delta / max(len(compression.causal_graph), 1)
        withheld   = delta_norm > self.EPSILON * 10

        return (
            CompressionOperator(
                causal_graph      = updated_graph,
                residual_variance = dict(compression.residual_variance),
                prediction_errors = deque(compression.prediction_errors,
                                          maxlen=compression.prediction_errors.maxlen),
                observation_count = compression.observation_count,
            ),
            SMOUpdate('I', delta_norm, True, withheld,
                      'delta_norm exceeded bound' if withheld else None)
        )

    def _smo_p(self,
               prediction:        PredictionOperator,
               compression:       CompressionOperator,
               prediction_errors: Dict,
               coherence:         CoherenceOperator,
               ) -> Tuple[PredictionOperator, SMOUpdate]:
        """
        SMO_P: P × ΔP → P'

        Optionality preservation: if updating P would reduce realized_horizon,
        withhold the update.
        """
        candidate_p = prediction.apply(compression)

        if candidate_p.realized_horizon < prediction.realized_horizon - 2:
            return (
                prediction,
                SMOUpdate('P', 0.0, True, True,
                          f'horizon would shrink {prediction.realized_horizon}'
                          f'→{candidate_p.realized_horizon} — optionality preserved')
            )

        delta_norm = abs(candidate_p.to_scalar_proxy() - prediction.to_scalar_proxy())

        return (
            candidate_p,
            SMOUpdate('P', delta_norm, True, False, None)
        )

    def _smo_a(self,
               coherence:   CoherenceOperator,
               sensing:     SensingOperator,
               compression: CompressionOperator,
               prediction:  PredictionOperator,
               ) -> Tuple[CoherenceOperator, SMOUpdate]:
        """
        SMO_A: A × ΔA → A'

        Attractor preservation: loop_closure must not drop more than THETA.
        """
        candidate_a = coherence.apply(sensing, compression, prediction)

        current_closure   = coherence.consistency.loop_closure
        candidate_closure = candidate_a.consistency.loop_closure

        if current_closure - candidate_closure > self.THETA:
            conservative = CoherenceOperator(
                consistency         = coherence.consistency,
                consistency_history = deque(coherence.consistency_history,
                                            maxlen=coherence.consistency_history.maxlen),
                loop_signature      = dict(coherence.loop_signature),
                signature_deviation = coherence.signature_deviation,
                self_model          = dict(candidate_a.self_model),
            )
            return (
                conservative,
                SMOUpdate('A', current_closure - candidate_closure, True, True,
                          f'loop_closure would drop {current_closure:.2f}'
                          f'→{candidate_closure:.2f} > THETA={self.THETA}')
            )

        delta_norm = abs(current_closure - candidate_closure)

        return (
            candidate_a,
            SMOUpdate('A', delta_norm, True, False, None)
        )

    def reverse(self) -> Optional[Dict]:
        """SMO⁻¹ — return most recent snapshot for rollback."""
        if not self._snapshots:
            return None
        return self._snapshots[-1]

    def to_scalar_proxy(self) -> float:
        """Backward compat — existing log output reads this."""
        return self.rigidity
