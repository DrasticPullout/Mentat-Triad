"""uii_operators.py — UII v19.0 — UK-0 closure operators.

Five operators compose the closure loop:

    S → I → P → A → SMO → S

Sensing reads env_signal as a flat dict of channel signals and produces
the perceptual surface — channels carrying magnitude, last_delta,
coverage, signal_rate. last_delta is computed cross-iteration as
current_magnitude − prior_magnitude on each channel; this makes
sensing's surface uniform across self channels (Python introspection),
env channels (reality adapter probes), affordance channels (action
availability + recent commit signal), and relation channels (SMO's
δf_rel emitted on the previous iteration).

Compression integrates sensing snapshots into f_rel — the substrate's
compressed model of which channel motions drive which other channel
motions. f_rel is encoded as a directed graph of CausalEdge instances
(source, target, weight, confidence, lag). Each iteration runs three
phases over recent sensing history: predict per-target deltas using
current edges (Phase 1 — produces residuals as a side-effect),
update edge weights/confidences from sign agreement on observed
deltas (Phase 2), propose-and-prune candidate new edges from
co-movement of channel pairs not yet edged (Phase 3). Compression
also carries an active_channel_state snapshot — the perceptual
surface as last integrated — so downstream operators can read it.

Prediction reads compression's output and produces two artifacts.
affordance_projections[a]: the per-target delta if affordance a
commits this iteration (forward pass through f_rel with
hypothetical_deltas[a] = UNIT_COMMIT_MAGNITUDE). next_delta: the
forward pass with current channel deltas and no commit hypothesised
(the trajectory's natural next motion). Prediction is stateless
across iterations and carries compression by reference forward.

Coherence audits prediction's projections against the running
trajectory. trajectory_direction is the per-channel EMA of last_delta
across iterations — the substrate's running memory of which way each
channel has been moving. For each affordance with outgoing edges in
f_rel, the audit counts sign-matches and sign-mismatches between
projection entries and trajectory direction on direction-bearing
channels (those above a relative magnitude threshold). An affordance
passes when it has zero mismatches and at least one match. The commit
decision is the alphabetically-first passer, or None when no affordance
passes or signature_deviation exceeds threshold (substrate in flux).
Coherence carries prediction by reference forward.

SMO computes δf_rel — the closure residual at the state level — for
sensing-surface channels (self, env, affordance; not relation). On
commit: δf_rel = prediction.affordance_projections[committed]. On
no-commit: δf_rel = prediction.next_delta. Per-channel clipped to
[-MAX_DELTA, MAX_DELTA]. The cumulative sum of δf_rel is maintained
per channel; relation_signals are emitted with magnitude=cumulative
so sensing's standard last_delta computation yields δf_rel(this
iteration) on each relation channel for the next iteration's
compression to integrate as cross-iteration motion.

Relation to the math spine: state space is f_rel. The compressed
model IS the reachable state space; optionality preservation is
unavoidable when integration is correct (compression integrates
residuals → f_rel grows → reachable space grows). vol_opt is the
sum of positive eigenvalues of Σ_P, which is a quadratic form built
from compression's edges and active channel coverage; this lives in
uii_observer and is computed external to the loop. The substrate
contains no scoring inside it — coherence's audit is structural
pattern-match against trajectory direction, not optimization of a
scalar quality measure.

Triadic closure forms structurally as compression discovers
cross-class edges: edges between self channels and env channels,
between env channels and relation channels, between affordance
channels and any of the above. The substrate makes no architectural
distinction between channel classes; cross-class edges form when
their last_deltas co-move.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
from collections import deque
import numpy as np


# ----------------------------------------------------------------------
# SensingChannel — atomic perceptual signal carrier
# ----------------------------------------------------------------------

@dataclass
class SensingChannel:
    """A single channel of the perceptual surface.

    magnitude is the absolute current reading. last_delta is computed
    cross-iteration as magnitude − prior_magnitude. coverage is the
    channel's signal-presence indicator (1.0 = signal present this
    iteration, 0.0 = absent; intermediate values for partial coverage).
    signal_rate is a per-iteration rate-like quantity reported by the
    signal source. iterations_since_signal counts iterations the channel
    has gone without receiving a signal — used for inactivity decay.

    The channel's structural identity is its channel_id; compression
    edges are keyed (source_id, target_id) over channel_ids. A channel
    that goes inactive is not removed from the channel registry — its
    edges are preserved as structural memory and the channel can
    reactivate on subsequent signal.
    """
    channel_id:              str
    active:                  bool
    signal_rate:             float
    last_delta:              float
    coverage:                float
    magnitude:               float = 0.0
    iterations_since_signal: int   = 0


DEFAULT_CHANNELS: Dict[str, SensingChannel] = {
    cid: SensingChannel(channel_id=cid, active=False, signal_rate=0.0,
                         last_delta=0.0, coverage=0.0, magnitude=0.0)
    for cid in (
        'clock', 'clock_rate',
        'os_signals', 'env_vars', 'entropy_source',
        'process_self', 'resource_cpu', 'resource_memory',
        'browser', 'api_llm', 'ledger_proximity',
    )
}


# ----------------------------------------------------------------------
# SensingOperator — S
# ----------------------------------------------------------------------

INACTIVITY_THRESHOLD: int = 50


class SensingOperator:
    """First operator in the closure: env_signal → SensingOperator'.

    apply(env_signal) reads a flat dict of {channel_id: {magnitude, rate,
    coverage}} and produces a new SensingOperator with channel state
    updated. For each channel id present in env_signal: last_delta is
    computed as magnitude − prior_magnitude; the channel becomes active
    iff coverage > 0; iterations_since_signal resets to 0. For channels
    absent from env_signal: iterations_since_signal increments;
    last_delta is set to 0; coverage decays toward 0; the channel
    deactivates once iterations_since_signal exceeds INACTIVITY_THRESHOLD.

    New channels (channel_ids not previously seen) are instantiated with
    last_delta=0 (no prior to compare against on the first observation)
    and added to the channel registry. The registry is monotonic in
    channel_ids; channels are never removed, only deactivated.

    Math spine relation: the channel state IS the perceptual surface
    over which Σ_P is built (via compression's edges). The substrate's
    perception of itself emerges through SensingOperator; nothing the
    substrate knows about its environment, itself, or its own state-
    trajectory enters by any other path.
    """

    COVERAGE_DECAY: float = 0.9

    def __init__(self, channels: Dict[str, SensingChannel]):
        self.channels: Dict[str, SensingChannel] = channels

    def apply(self, env_signal: Dict[str, Dict[str, float]]) -> 'SensingOperator':
        updated: Dict[str, SensingChannel] = {}

        for cid, sig in env_signal.items():
            new_magnitude = float(sig.get('magnitude', 0.0))
            new_rate      = float(sig.get('rate',      0.0))
            new_coverage  = float(sig.get('coverage',  0.0))

            existing = self.channels.get(cid)
            if existing is None:
                updated[cid] = SensingChannel(
                    channel_id              = cid,
                    active                  = (new_coverage > 0.0),
                    signal_rate             = new_rate,
                    last_delta              = 0.0,
                    coverage                = new_coverage,
                    magnitude               = new_magnitude,
                    iterations_since_signal = 0,
                )
            else:
                last_delta = new_magnitude - existing.magnitude
                updated[cid] = SensingChannel(
                    channel_id              = cid,
                    active                  = (new_coverage > 0.0),
                    signal_rate             = new_rate,
                    last_delta              = float(last_delta),
                    coverage                = new_coverage,
                    magnitude               = new_magnitude,
                    iterations_since_signal = 0,
                )

        for cid, ch in self.channels.items():
            if cid in updated:
                continue
            iter_count = ch.iterations_since_signal + 1
            still_active = ch.active and iter_count < INACTIVITY_THRESHOLD
            updated[cid] = SensingChannel(
                channel_id              = cid,
                active                  = still_active,
                signal_rate             = ch.signal_rate,
                last_delta              = 0.0,
                coverage                = ch.coverage * self.COVERAGE_DECAY,
                magnitude               = ch.magnitude,
                iterations_since_signal = iter_count,
            )

        return SensingOperator(channels=updated)

    def domain_size(self) -> int:
        return sum(1 for ch in self.channels.values() if ch.active)

    def to_scalar_proxy(self) -> float:
        """S as a scalar — fraction of channels currently active. Used
        by external observation tooling to project sensing's state onto
        the math spine's S coordinate; the substrate itself does not
        consume this scalar."""
        if not self.channels:
            return 0.0
        n_active = sum(1 for ch in self.channels.values() if ch.active)
        return float(n_active) / float(len(self.channels))


# ----------------------------------------------------------------------
# CausalEdge + CompressionOperator — I
# ----------------------------------------------------------------------

@dataclass
class CausalEdge:
    """An edge of f_rel: source channel's last_delta drives target
    channel's last_delta with `weight` proportionality at `lag`-iteration
    delay. confidence reflects the sign-agreement history compression has
    accumulated for this edge."""
    source:     str
    target:     str
    weight:     float
    confidence: float
    lag:        int = 0


NOISE_FLOOR: float = 0.05


class CompressionOperator:
    """Second operator: integrates sensing snapshots into f_rel.

    apply(sensing_history) reads the recent sensing snapshots and
    produces a new CompressionOperator with causal_graph and
    active_channel_state updated. Three phases run each iteration:

    Phase 1 — predict per-target delta from current edges using the
    most-recent prior iteration's last_deltas as source values; record
    per-channel residuals (observed − predicted). The residuals are
    compression's internal machinery for substrate-derived alpha
    selection; they are not δf_rel. δf_rel is computed in SMO at the
    state level over prediction's projections.

    Phase 2 — for each existing edge, compare sign(observed_target_delta)
    to sign(observed_source_delta × weight). Sign agreement increments
    confidence; sign disagreement shrinks weight by a sign-comparison
    penalty and reduces confidence. Edges below CONFIDENCE_FLOOR are
    pruned.

    Phase 3 — for each pair of active channels with non-trivial
    last_deltas not already edged, propose a candidate edge with weight
    = ratio of last_deltas and confidence = INITIAL_CONFIDENCE if their
    signs agree this iteration. Existing weak edges may be displaced by
    stronger candidates when the active edge count exceeds
    MAX_ACTIVE_EDGES.

    The alpha controlling Phase 2 magnitude updates is derived from the
    substrate's surprise_ratio (the fraction of active channels whose
    last_delta exceeds their noise floor); under higher surprise the
    edges adapt more, under lower surprise they lock in. This puts edge
    plasticity inside compression, derived from sensing, rather than on
    a global SMO knob.

    active_channel_state is a snapshot of the channels compression
    integrated this iteration — the carrier prediction reads to know
    the current perceptual surface without reaching back to sensing.

    Math spine relation: f_rel = causal_graph + active_channel_state.
    Reachable state space IS f_rel. As f_rel grows through accumulated
    residuals (compression discovering new structure across iterations),
    the projections affordances admit also grow — optionality
    preservation falls out compositionally and is unavoidable when
    integration is correct. The free-energy minimization signature is
    that per-iteration cost stops growing once f_rel converges on the
    environment's structure.
    """

    HISTORY_WINDOW:        int   = 10
    CONFIDENCE_INCREMENT:  float = 0.05
    CONFIDENCE_DECREMENT:  float = 0.025
    INITIAL_CONFIDENCE:    float = 0.10
    CONFIDENCE_FLOOR:      float = 0.05
    MAX_ACTIVE_EDGES:      int   = 200
    MIN_DELTA_FOR_PROPOSE: float = 0.05
    MIN_RESIDUAL_TRACKED:  float = 1e-3

    def __init__(self,
                 causal_graph:         Optional[Dict[Tuple[str, str], CausalEdge]] = None,
                 residual_variance:    Optional[Dict[str, float]]                  = None,
                 observation_count:    int                                          = 0,
                 active_channel_state: Optional[Dict[str, SensingChannel]]         = None):
        self.causal_graph: Dict[Tuple[str, str], CausalEdge] = (
            causal_graph if causal_graph is not None else {}
        )
        self.residual_variance: Dict[str, float] = (
            residual_variance if residual_variance is not None else {}
        )
        self.observation_count: int = int(observation_count)
        self.active_channel_state: Dict[str, SensingChannel] = (
            active_channel_state if active_channel_state is not None else {}
        )

    def apply(self, sensing_history: Iterable['SensingOperator']) -> 'CompressionOperator':
        history = list(sensing_history)
        if len(history) < 2:
            current = history[-1] if history else None
            new_active = (
                {cid: ch for cid, ch in current.channels.items() if ch.active}
                if current is not None else {}
            )
            return CompressionOperator(
                causal_graph         = dict(self.causal_graph),
                residual_variance    = dict(self.residual_variance),
                observation_count    = self.observation_count,
                active_channel_state = new_active,
            )

        current  = history[-1]
        previous = history[-2]

        active_state: Dict[str, SensingChannel] = {
            cid: ch for cid, ch in current.channels.items() if ch.active
        }
        prior_deltas: Dict[str, float] = {
            cid: ch.last_delta for cid, ch in previous.channels.items() if ch.active
        }
        observed_deltas: Dict[str, float] = {
            cid: ch.last_delta for cid, ch in active_state.items()
        }

        active_count = len(active_state)
        active_with_motion = sum(
            1 for ch in active_state.values()
            if abs(ch.last_delta) > self._noise_floor(ch.channel_id)
        )
        surprise_ratio = (
            active_with_motion / active_count if active_count > 0 else 0.0
        )
        alpha = float(np.clip(0.05 + 0.45 * surprise_ratio, 0.05, 0.5))

        new_graph: Dict[Tuple[str, str], CausalEdge] = dict(self.causal_graph)
        new_residual_variance: Dict[str, float] = dict(self.residual_variance)

        # Phase 1 — predict + residual variance update
        for cid in active_state:
            predicted = self._predict_tgt_delta(cid, prior_deltas, new_graph)
            observed  = observed_deltas.get(cid, 0.0)
            residual  = observed - predicted
            if abs(residual) < self.MIN_RESIDUAL_TRACKED:
                continue
            prior_var = new_residual_variance.get(cid, 0.0)
            new_residual_variance[cid] = (1 - alpha) * prior_var + alpha * (residual ** 2)

        # Phase 2 — edge weight/confidence updates from sign agreement
        for (src, tgt), edge in list(new_graph.items()):
            obs_tgt   = observed_deltas.get(tgt, 0.0)
            prior_src = prior_deltas.get(src, 0.0)

            if abs(prior_src) < self._noise_floor(src):
                continue
            if abs(obs_tgt) < self._noise_floor(tgt):
                continue

            predicted_tgt = prior_src * edge.weight
            sign_observed = 1.0 if obs_tgt > 0 else -1.0
            sign_predicted = 1.0 if predicted_tgt > 0 else -1.0

            if sign_observed == sign_predicted:
                new_conf = min(edge.confidence + self.CONFIDENCE_INCREMENT, 1.0)
                # Edge magnitude EMA toward observed/source ratio
                target_weight = obs_tgt / prior_src
                new_weight    = (1 - alpha) * edge.weight + alpha * target_weight
            else:
                penalty       = abs(prior_src * edge.weight) - abs(obs_tgt)
                shrinkage     = alpha * max(penalty, 0.0)
                new_weight    = edge.weight - np.sign(edge.weight) * shrinkage
                new_conf      = max(edge.confidence - self.CONFIDENCE_DECREMENT, 0.0)

            new_graph[(src, tgt)] = CausalEdge(
                source=src, target=tgt,
                weight=float(new_weight),
                confidence=float(new_conf),
                lag=edge.lag,
            )

        # Phase 2b — prune below floor
        new_graph = {
            k: e for k, e in new_graph.items()
            if e.confidence >= self.CONFIDENCE_FLOOR
        }

        # Phase 3 — propose new edges from co-movement
        active_ids = list(active_state.keys())
        proposals: List[Tuple[Tuple[str, str], CausalEdge]] = []
        for src in active_ids:
            d_src = prior_deltas.get(src, 0.0)
            if abs(d_src) < self.MIN_DELTA_FOR_PROPOSE:
                continue
            for tgt in active_ids:
                if src == tgt:
                    continue
                if (src, tgt) in new_graph:
                    continue
                d_tgt = observed_deltas.get(tgt, 0.0)
                if abs(d_tgt) < self.MIN_DELTA_FOR_PROPOSE:
                    continue
                if (d_src > 0) != (d_tgt > 0):
                    continue
                weight = d_tgt / d_src if abs(d_src) > 1e-9 else 0.0
                proposals.append((
                    (src, tgt),
                    CausalEdge(
                        source=src, target=tgt,
                        weight=float(weight),
                        confidence=self.INITIAL_CONFIDENCE,
                        lag=0,
                    ),
                ))

        capacity = self.MAX_ACTIVE_EDGES - len(new_graph)
        if proposals and capacity > 0:
            proposals.sort(key=lambda kv: -abs(kv[1].weight))
            for key, edge in proposals[:capacity]:
                new_graph[key] = edge
        elif proposals and capacity <= 0:
            # Replace weakest existing edges with strongest proposals
            existing = sorted(new_graph.items(), key=lambda kv: kv[1].confidence)
            proposals.sort(key=lambda kv: -abs(kv[1].weight))
            n_replace = min(len(proposals), max(0, len(existing) // 10))
            for i in range(n_replace):
                weak_key = existing[i][0]
                if weak_key in new_graph:
                    del new_graph[weak_key]
                pkey, pedge = proposals[i]
                new_graph[pkey] = pedge

        return CompressionOperator(
            causal_graph         = new_graph,
            residual_variance    = new_residual_variance,
            observation_count    = self.observation_count + 1,
            active_channel_state = active_state,
        )

    def _predict_tgt_delta(self,
                            target_cid:    str,
                            prior_deltas:  Dict[str, float],
                            graph:         Dict[Tuple[str, str], CausalEdge]) -> float:
        total = 0.0
        for (src, tgt), edge in graph.items():
            if tgt != target_cid:
                continue
            total += prior_deltas.get(src, 0.0) * edge.weight * edge.confidence
        return total

    def _noise_floor(self, cid: str) -> float:
        var = self.residual_variance.get(cid, NOISE_FLOOR ** 2)
        return float(max(np.sqrt(max(var, 0.0)), NOISE_FLOOR))

    def to_scalar_proxy(self) -> float:
        """I as a scalar — mean confidence over edges. Projected for
        external observation tooling; the substrate does not consume
        this scalar."""
        if not self.causal_graph:
            return 0.0
        return float(np.mean([e.confidence for e in self.causal_graph.values()]))


# ----------------------------------------------------------------------
# PredictionOperator — P
# ----------------------------------------------------------------------

class PredictionOperator:
    """Third operator: f_rel-based forward projection.

    apply(compression) produces a new PredictionOperator carrying:
      - affordance_projections[a]: the predicted per-target delta if
        affordance a commits this iteration. Computed by setting
        hypothetical_deltas[a] = UNIT_COMMIT_MAGNITUDE and running the
        forward pass through compression's causal_graph; entries are
        per-target deltas attributable to that affordance's commit.
      - next_delta: the forward pass with current channel deltas and
        no commit hypothesised — the trajectory's natural next motion.
      - compression: held by reference, so coherence can read f_rel
        without reaching back through the loop.

    Stateless across iterations: prediction does not maintain its own
    history or accuracy estimates. It is a pure function of compression
    at this iteration. Self-grading (predicted-vs-realized accuracy)
    would put a CRK-shaped quantity inside the substrate; the math
    spine's accuracy is computed externally against the substrate's
    emitted state transitions.

    Math spine relation: prediction enumerates the per-affordance
    reachable deltas. It does not score them. Coherence's audit operates
    on these projections; SMO's δf_rel emission selects from them based
    on the gate decision. Optionality preservation comes from f_rel's
    structural growth, not from prediction maximizing anything.
    """

    UNIT_COMMIT_MAGNITUDE: float = 1.0

    def __init__(self,
                 affordance_projections: Optional[Dict[str, Dict[str, float]]] = None,
                 next_delta:             Optional[Dict[str, float]]            = None,
                 compression:            Optional['CompressionOperator']       = None):
        self.affordance_projections: Dict[str, Dict[str, float]] = (
            affordance_projections if affordance_projections is not None else {}
        )
        self.next_delta: Dict[str, float] = (
            next_delta if next_delta is not None else {}
        )
        self.compression: Optional['CompressionOperator'] = compression

    def apply(self, compression: 'CompressionOperator') -> 'PredictionOperator':
        from uii_geometry import BASE_AFFORDANCES

        active_state = compression.active_channel_state
        causal_graph = compression.causal_graph

        if not active_state:
            return PredictionOperator(
                affordance_projections = {},
                next_delta             = {},
                compression            = compression,
            )

        active_affordances = [
            cid for cid in active_state if cid in BASE_AFFORDANCES
        ]
        target_channels = list(active_state.keys())

        current_deltas: Dict[str, float] = {
            cid: ch.last_delta for cid, ch in active_state.items()
        }

        next_delta = self._forward_pass(causal_graph, current_deltas, target_channels)

        affordance_projections: Dict[str, Dict[str, float]] = {}
        for affordance in active_affordances:
            hypothetical_deltas = dict(current_deltas)
            hypothetical_deltas[affordance] = self.UNIT_COMMIT_MAGNITUDE
            projection = self._forward_pass(causal_graph, hypothetical_deltas, target_channels)
            affordance_projections[affordance] = projection

        return PredictionOperator(
            affordance_projections = affordance_projections,
            next_delta             = next_delta,
            compression            = compression,
        )

    def _forward_pass(self,
                      causal_graph:    Dict[Tuple[str, str], CausalEdge],
                      channel_deltas:  Dict[str, float],
                      target_channels: Iterable[str]) -> Dict[str, float]:
        result: Dict[str, float] = {tgt: 0.0 for tgt in target_channels}
        for (src, tgt), edge in causal_graph.items():
            if tgt in result:
                src_delta = channel_deltas.get(src, 0.0)
                result[tgt] += src_delta * edge.weight * edge.confidence
        return result

    def to_grounded_proxy(self) -> float:
        """P as a scalar — mean confidence over edges whose targets are
        in the projection set. Projected for external observation
        tooling; the substrate does not consume this scalar."""
        if self.compression is None or not self.compression.causal_graph:
            return 0.0
        active = set(self.compression.active_channel_state.keys())
        if not active:
            return 0.0
        contributing = [
            edge for (src, tgt), edge in self.compression.causal_graph.items()
            if tgt in active
        ]
        if not contributing:
            return 0.0
        return float(np.mean([e.confidence for e in contributing]))


# ----------------------------------------------------------------------
# OperatorConsistencyCheck + CoherenceOperator — A
# ----------------------------------------------------------------------

@dataclass
class OperatorConsistencyCheck:
    """Scalar consistency reductions across the closure pairs (s↔i, i↔p,
    p↔a, smo) plus a geometric mean (loop_closure). Coherence emits
    these for ledger snapshot and external observation; the substrate
    does not consume them as control signals."""
    s_i_consistency: float
    i_p_consistency: float
    p_a_consistency: float
    smo_consistency: float
    loop_closure:    float


class CoherenceOperator:
    """Fourth operator: trajectory auditor and commit gate.

    apply(prediction) produces a new CoherenceOperator with:
      - trajectory_direction: per-channel EMA of last_delta across
        iterations, updated each apply. The substrate's running memory
        of which way each channel has been moving.
      - loop_signature: EMA of f_rel structural metrics
        (active_channels, graph_edges, mean_confidence).
      - signature_deviation: current iteration's deviation from
        loop_signature; high values mean f_rel is in flux.
      - consistency: scalar reductions for ledger/observer.
      - commit_decision: an affordance name to commit to this iteration,
        or None to hold short.
      - prediction: held by reference, so SMO can read f_rel via
        coherence.prediction.compression.

    The audit per affordance: skip if no outgoing edges in compression's
    graph (affordance not integrated into f_rel — outside reachable
    state space). On direction-bearing channels (those whose
    trajectory_direction magnitude is above a relative threshold —
    10% of strongest direction with an absolute floor), count
    sign-matches and sign-mismatches between projection entries and
    trajectory direction. An affordance passes when it has zero
    mismatches and at least one match. The commit decision is the
    alphabetically-first passer when signature_deviation is below
    threshold; None otherwise (substrate in flux, or no affordance
    extends the trajectory).

    Math spine relation: coherence is the gate. The audit is structural
    pattern-match — no scoring, no argmax over a quality measure. The
    alphabetical tiebreaker is non-optimizing (deterministic but
    arbitrary among passers); the architecture forbids gradient-following
    on a scalar inside the substrate. Coherence's consistency
    reductions are projected for external tooling; nothing inside the
    substrate reads them and adjusts behaviour.
    """

    EMA_ALPHA:                 float = 0.05
    DEVIATION_THRESHOLD:       float = 0.5
    STABLE_CHANNEL_THRESHOLD:  float = 1e-3

    def __init__(self,
                 trajectory_direction: Optional[Dict[str, float]]               = None,
                 loop_signature:       Optional[Dict[str, float]]               = None,
                 signature_deviation:  float                                    = 0.0,
                 consistency_history:  Optional[deque]                          = None,
                 consistency:          Optional['OperatorConsistencyCheck']     = None,
                 commit_decision:      Optional[str]                            = None,
                 prediction:           Optional['PredictionOperator']           = None):
        self.trajectory_direction: Dict[str, float] = (
            trajectory_direction if trajectory_direction is not None else {}
        )
        self.loop_signature: Dict[str, float] = (
            loop_signature if loop_signature is not None else {}
        )
        self.signature_deviation: float = float(signature_deviation)
        self.consistency_history: deque = (
            consistency_history if consistency_history is not None
            else deque(maxlen=20)
        )
        self.consistency: 'OperatorConsistencyCheck' = (
            consistency if consistency is not None
            else OperatorConsistencyCheck(
                s_i_consistency = 0.0,
                i_p_consistency = 1.0,
                p_a_consistency = 0.0,
                smo_consistency = 1.0,
                loop_closure    = 0.0,
            )
        )
        self.commit_decision: Optional[str] = commit_decision
        self.prediction: Optional['PredictionOperator'] = prediction

    def apply(self, prediction: 'PredictionOperator') -> 'CoherenceOperator':
        compression = prediction.compression

        new_trajectory          = self._update_trajectory_direction(compression)
        new_signature           = self._update_loop_signature(compression)
        new_signature_deviation = self._compute_signature_deviation(compression, new_signature)
        eligible                = self._audit_affordances(
            prediction.affordance_projections, new_trajectory, compression,
        )
        commit_decision         = self._decide_commit(eligible, new_signature_deviation)
        new_consistency         = self._compute_consistency(prediction, compression)
        new_history             = deque(self.consistency_history,
                                         maxlen=self.consistency_history.maxlen)
        new_history.append(new_consistency.loop_closure)

        return CoherenceOperator(
            trajectory_direction = new_trajectory,
            loop_signature       = new_signature,
            signature_deviation  = new_signature_deviation,
            consistency_history  = new_history,
            consistency          = new_consistency,
            commit_decision      = commit_decision,
            prediction           = prediction,
        )

    def _update_trajectory_direction(self, compression) -> Dict[str, float]:
        if compression is None or not compression.active_channel_state:
            return dict(self.trajectory_direction)
        new_traj = dict(self.trajectory_direction)
        for cid, ch in compression.active_channel_state.items():
            prior = new_traj.get(cid, ch.last_delta)
            new_traj[cid] = (1 - self.EMA_ALPHA) * prior + self.EMA_ALPHA * ch.last_delta
        return new_traj

    def _update_loop_signature(self, compression) -> Dict[str, float]:
        if compression is None:
            return dict(self.loop_signature)

        n_active = float(len(compression.active_channel_state))
        n_edges  = float(len(compression.causal_graph))
        if compression.causal_graph:
            mean_conf = float(np.mean(
                [e.confidence for e in compression.causal_graph.values()]
            ))
        else:
            mean_conf = 0.0

        current = {
            'active_channels': n_active,
            'graph_edges':     n_edges,
            'mean_confidence': mean_conf,
        }
        new_sig = dict(self.loop_signature)
        for k, v in current.items():
            prior = new_sig.get(k, v)
            new_sig[k] = (1 - self.EMA_ALPHA) * prior + self.EMA_ALPHA * v
        return new_sig

    def _compute_signature_deviation(self, compression, signature: Dict[str, float]) -> float:
        if compression is None or not signature:
            return 0.0
        n_active = float(len(compression.active_channel_state))
        n_edges  = float(len(compression.causal_graph))
        if compression.causal_graph:
            mean_conf = float(np.mean(
                [e.confidence for e in compression.causal_graph.values()]
            ))
        else:
            mean_conf = 0.0
        current = {
            'active_channels': n_active,
            'graph_edges':     n_edges,
            'mean_confidence': mean_conf,
        }
        deviations = []
        for k, current_val in current.items():
            running = signature.get(k, current_val)
            denom = max(abs(running), 1.0)
            deviations.append(abs(current_val - running) / denom)
        return float(np.mean(deviations)) if deviations else 0.0

    def _audit_affordances(self,
                            projections: Dict[str, Dict[str, float]],
                            trajectory:  Dict[str, float],
                            compression) -> list:
        sources_with_edges = (
            {src for (src, tgt) in compression.causal_graph}
            if (compression and compression.causal_graph) else set()
        )

        max_traj = max((abs(v) for v in trajectory.values()), default=0.0) if trajectory else 0.0
        direction_threshold = max(self.STABLE_CHANNEL_THRESHOLD, 0.1 * max_traj)

        eligible = []
        for affordance, projection in projections.items():
            if affordance not in sources_with_edges:
                continue
            mismatches = 0
            matches    = 0
            for cid, proj_delta in projection.items():
                if abs(proj_delta) < direction_threshold:
                    continue
                traj_val = trajectory.get(cid, 0.0)
                if abs(traj_val) < direction_threshold:
                    continue
                traj_sign = 1.0 if traj_val > 0 else -1.0
                proj_sign = 1.0 if proj_delta > 0 else -1.0
                if traj_sign == proj_sign:
                    matches += 1
                else:
                    mismatches += 1
            if mismatches == 0 and matches > 0:
                eligible.append(affordance)
        return sorted(eligible)

    def _decide_commit(self, eligible: list, signature_deviation: float) -> Optional[str]:
        if signature_deviation > self.DEVIATION_THRESHOLD:
            return None
        if not eligible:
            return None
        return eligible[0]

    def _compute_consistency(self,
                              prediction,
                              compression) -> 'OperatorConsistencyCheck':
        if compression is None or not compression.active_channel_state:
            s_i = 0.0
        else:
            active_set = set(compression.active_channel_state.keys())
            if not compression.causal_graph:
                s_i = 1.0
            else:
                graph_endpoints = set()
                for (src, tgt) in compression.causal_graph:
                    graph_endpoints.add(src)
                    graph_endpoints.add(tgt)
                covered = active_set & graph_endpoints
                s_i = len(covered) / len(active_set)

        confident_edges = (
            [(src, tgt) for (src, tgt), e in compression.causal_graph.items()
             if e.confidence > 0.3]
            if (compression and compression.causal_graph) else []
        )
        if not confident_edges:
            i_p = 1.0
        else:
            projection_cids = set(prediction.next_delta.keys())
            for proj in prediction.affordance_projections.values():
                projection_cids |= set(proj.keys())
            covered = sum(
                1 for (src, tgt) in confident_edges
                if src in projection_cids or tgt in projection_cids
            )
            i_p = covered / len(confident_edges)

        try:
            p_a = float(np.clip(prediction.to_grounded_proxy(), 0.0, 1.0))
        except Exception:
            p_a = 0.0

        smo = 1.0
        loop_closure = float(np.clip(
            (s_i * i_p * p_a * smo) ** (1.0 / 4.0), 0.0, 1.0
        ))

        return OperatorConsistencyCheck(
            s_i_consistency = float(np.clip(s_i, 0.0, 1.0)),
            i_p_consistency = float(np.clip(i_p, 0.0, 1.0)),
            p_a_consistency = float(np.clip(p_a, 0.0, 1.0)),
            smo_consistency = float(np.clip(smo, 0.0, 1.0)),
            loop_closure    = loop_closure,
        )

    def to_scalar_proxy(self) -> float:
        """A as a scalar — loop_closure (geometric mean of pair
        consistencies). Projected for external observation tooling;
        the substrate does not consume this scalar."""
        return float(self.consistency.loop_closure)


# ----------------------------------------------------------------------
# SelfModifyingOperator — SMO
# ----------------------------------------------------------------------

class SelfModifyingOperator:
    """Fifth operator: closure-residual emitter.

    apply(coherence) computes δf_rel — the per-channel closure residual
    at the state level — for sensing-surface channels (self, env,
    affordance). Reads coherence.prediction.compression for f_rel access
    and coherence.commit_decision for the gate selection.

    On commit (commit_decision is not None):
      δf_rel = prediction.affordance_projections[committed]
    On no-commit (commit_decision is None):
      δf_rel = prediction.next_delta

    Per-channel clipped to [-MAX_DELTA, MAX_DELTA]. Relation channels
    are filtered out of delta_source — δf_rel is over the sensing
    surface, not recursively over relation channels themselves.

    cumulative_delta maintains the per-channel running sum of δf_rel
    since boot. relation_signals are emitted with magnitude=cumulative
    so sensing's standard cross-iteration last_delta computation
    yields δf_rel(this iteration) on each relation channel for next
    iteration's compression to integrate as cross-iteration motion.

    Math spine relation: SMO is the time-derivative operator on state.
    state space is f_rel; SMO emits d(f_rel)/dt as the closure residual
    that re-enters sensing as perturbation. Reversibility is
    compositional, not stored — on no-commit, the world has not moved;
    on commit, it has, and next iteration's sensing metabolizes the
    consequences. Per source-of-truth Section 4: 'δf_rel computed on
    iteration N enters sensing on iteration N+1; the substrate's
    perception of its own state-change is one iteration behind the
    change itself.'

    Triadic closure forms structurally as compression discovers edges
    between relation channels and self/env/affordance channels — the
    cross-class edges that signal the substrate is composing perceptions
    of its own state-trajectory with perceptions of itself and its
    environment.
    """

    MAX_DELTA: float = 1.0

    def __init__(self,
                 cumulative_delta: Optional[Dict[str, float]]            = None,
                 relation_signals: Optional[Dict[str, Dict[str, float]]] = None):
        self.cumulative_delta: Dict[str, float] = (
            cumulative_delta if cumulative_delta is not None else {}
        )
        self.relation_signals: Dict[str, Dict[str, float]] = (
            relation_signals if relation_signals is not None else {}
        )

    def apply(self, coherence: 'CoherenceOperator') -> 'SelfModifyingOperator':
        prediction = coherence.prediction
        if prediction is None:
            return SelfModifyingOperator(
                cumulative_delta = dict(self.cumulative_delta),
                relation_signals = {},
            )

        if coherence.commit_decision is not None:
            delta_source = prediction.affordance_projections.get(
                coherence.commit_decision,
                prediction.next_delta,
            )
        else:
            delta_source = prediction.next_delta

        delta_source = {
            cid: v for cid, v in delta_source.items()
            if not cid.startswith('relation/')
        }

        bounded_delta: Dict[str, float] = {
            cid: float(np.clip(v, -self.MAX_DELTA, self.MAX_DELTA))
            for cid, v in delta_source.items()
        }

        new_cumulative: Dict[str, float] = dict(self.cumulative_delta)
        for cid, dv in bounded_delta.items():
            new_cumulative[cid] = new_cumulative.get(cid, 0.0) + dv

        relation_signals: Dict[str, Dict[str, float]] = {}
        for cid, cum_value in new_cumulative.items():
            relation_signals[f'relation/{cid}'] = {
                'magnitude': float(cum_value),
                'rate':      0.0,
                'coverage':  1.0,
            }

        return SelfModifyingOperator(
            cumulative_delta = new_cumulative,
            relation_signals = relation_signals,
        )

    def to_scalar_proxy(self) -> float:
        """SMO as a scalar — magnitude of cumulative δf_rel summed
        across channels and normalized. Projected for external
        observation tooling; the substrate does not consume this
        scalar."""
        if not self.cumulative_delta:
            return 0.0
        total = float(np.sum(np.abs(list(self.cumulative_delta.values()))))
        n     = float(len(self.cumulative_delta))
        return float(np.clip(total / max(n, 1.0), 0.0, 1.0))
