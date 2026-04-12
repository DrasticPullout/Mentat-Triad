"""
uii_ledger.py — v16
Persistence mechanism for the best attractor basin found across runs.

Not memory. Not lineage. A geometric description of the most stable
configuration the field has reached so far.

Multiple independent triads can load the same Ledger and converge on the
same attractor basin because the basin exists in the field geometry, not
in any particular instance.

Contents:
  - TriadLedger            (dataclass — persisted basin geometry)
  - PeakOptionalityTracker (writes ledger snapshot at peak Vol_opt)
  - load_ledger()          (pure deserialization — no mutation, no momentum)
  - save_ledger()          (serialize to JSON)
  - _snapshot_operators()  (private — serializes operator state)
  - _json_default()        (JSON serialization fallback)

Imports from uii_operators only. No circular dependencies.
"""

from __future__ import annotations

import copy
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque

from uii_operators import (
    SensingOperator,
    SensingChannel,
    CompressionOperator,
    CausalEdge,
    PredictionOperator,
    ChannelPrediction,
    CoherenceOperator,
    OperatorConsistencyCheck,
    SelfModifyingOperator,
    DEFAULT_CHANNELS,
)


# ──────────────────────────────────────────────────────────────────────────────
# JSON serialization helper
# ──────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    """Custom JSON serializer for types json.dumps can't handle natively."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


# ──────────────────────────────────────────────────────────────────────────────
# TriadLedger
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TriadLedger:
    """
    Geometric description of the best attractor basin found across runs.

    Three independent concerns stored separately so they can be written
    at different times and by different owners:

      hessian_snapshot   — written by PeakOptionalityTracker at peak Vol_opt
      operator_snapshot  — written by PeakOptionalityTracker at peak Vol_opt
      causal_model       — written by FAO.distill_to_ledger() at session end
      discovered_structure — written by FAO.distill_to_ledger() at session end

    hessian_snapshot and operator_snapshot are NEVER overwritten at session
    end — session-end state may be degraded. PeakOptionalityTracker owns them.
    """

    # ── Hessian snapshot at peak Vol_opt ──────────────────────────────────────
    hessian_snapshot: Dict = field(default_factory=dict)
    # Keys:
    #   matrix:        List[List[float]]  — full n×n H matrix
    #   eigenvalues:   List[float]        — ascending spectrum (eigh order)
    #   eigenvectors:  List[List[float]]  — full n×n eigenvector matrix
    #   vol_opt:       float              — Σλᵢ where λᵢ > 0 (comparison gate)
    #   phi:           float              — Φ value at this snapshot
    #   channel_basis: Dict              — coordinate system (see below)
    #
    # channel_basis keys:
    #   ordering:  List[str]             — ordered list of active channel IDs
    #                                      defines axis ↔ index correspondence
    #   metadata:  Dict[str, Dict]       — {channel_id: {active, coverage, signal_rate}}
    #   n_dims:    int                   — len(ordering)

    # ── Operator state at peak Vol_opt ────────────────────────────────────────
    operator_snapshot: Dict = field(default_factory=dict)
    # Keys match the full operator state needed to reconstruct each operator.
    #
    # sensing:
    #   channels: {channel_id: {active, coverage, signal_rate, last_delta}}
    #
    # compression:
    #   causal_graph: {"src,tgt": {weight, confidence, lag}}
    #     NOTE: tuple keys are serialized as "src,tgt" strings for JSON compat
    #   residual_variance: {channel_id: float}
    #   observation_count: int
    #
    # prediction:
    #   channel_predictions: {channel_id: {predicted_delta, confidence, horizon}}
    #   prediction_accuracy: {channel_id: float}
    #   realized_horizon: int
    #
    # coherence:
    #   loop_signature: {key: float}
    #   signature_deviation: float
    #   consistency: {s_i, i_p, p_a, smo, loop_closure}
    #
    # smo:
    #   plasticity: float
    #   rigidity:   float

    # ── Learned causal model ──────────────────────────────────────────────────
    causal_model: Dict = field(default_factory=dict)
    # Keys:
    #   coupling_matrix:      {matrix: List[List], confidence: float, observations: int}
    #   action_substrate_map: {action_str: {S, I, P, A}}
    #   migration_geometry:   {bad_hashes, successful_coupling_states,
    #                          outcome_counts, total_attempts}  (optional)

    # ── Discovered structure (Layer 3) ────────────────────────────────────────
    discovered_structure: Dict = field(default_factory=dict)
    # Two-tier lifecycle (provisional / admitted) managed by AxisAdmissionTest
    # in uii_fao.py. Ledger is the persistence medium; FAO owns lifecycle logic.


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _snapshot_operators(state) -> Dict:
    """
    Serialize all five operators to JSON-safe dicts.
    Called only by PeakOptionalityTracker.update().

    Tuple keys in causal_graph are serialized as "src,tgt" strings.
    state: SubstrateState — must expose .sensing, .compression,
           .prediction, .coherence attributes plus smo_v151 is
           passed separately; we read it from state.smo (scalar compat shim)
           for plasticity/rigidity values only.
    """
    s = state.sensing
    i = state.compression
    p = state.prediction
    a = state.coherence

    # ── sensing ───────────────────────────────────────────────────────────────
    sensing_snap = {
        'channels': {
            cid: {
                'active':      ch.active,
                'coverage':    float(ch.coverage),
                'signal_rate': float(ch.signal_rate),
                'last_delta':  float(ch.last_delta),
            }
            for cid, ch in s.channels.items()
        }
    }

    # ── compression ───────────────────────────────────────────────────────────
    # Tuple keys → "src,tgt" strings
    causal_graph_serial = {
        f"{src},{tgt}": {
            'weight':     float(edge.weight),
            'confidence': float(edge.confidence),
            'lag':        int(edge.lag),
        }
        for (src, tgt), edge in i.causal_graph.items()
    }
    compression_snap = {
        'causal_graph':      causal_graph_serial,
        'residual_variance': {k: float(v) for k, v in i.residual_variance.items()},
        'observation_count': int(i.observation_count),
    }

    # ── prediction ────────────────────────────────────────────────────────────
    channel_preds_serial = {
        cid: {
            'predicted_delta': float(cp.predicted_delta),
            'confidence':      float(cp.confidence),
            'horizon':         int(cp.horizon),
        }
        for cid, cp in p.channel_predictions.items()
    }
    prediction_snap = {
        'channel_predictions': channel_preds_serial,
        'prediction_accuracy': {k: float(v) for k, v in p.prediction_accuracy.items()},
        'realized_horizon':    int(p.realized_horizon),
    }

    # ── coherence ─────────────────────────────────────────────────────────────
    cons = a.consistency
    coherence_snap = {
        'loop_signature':      {k: float(v) for k, v in a.loop_signature.items()},
        'signature_deviation': float(a.signature_deviation),
        'consistency': {
            's_i':          float(cons.s_i_consistency),
            'i_p':          float(cons.i_p_consistency),
            'p_a':          float(cons.p_a_consistency),
            'smo':          float(cons.smo_consistency),
            'loop_closure': float(cons.loop_closure),
        },
    }

    # ── smo ───────────────────────────────────────────────────────────────────
    # Read plasticity/rigidity from the scalar compat shim on SubstrateState.
    # If smo_v151 is attached to state, prefer it; fall back to state.smo.
    smo_obj = getattr(state, 'smo_v151', None) or getattr(state, 'smo', None)
    smo_snap = {
        'plasticity': float(getattr(smo_obj, 'plasticity', 0.5)),
        'rigidity':   float(getattr(smo_obj, 'rigidity',   0.5)),
    }

    return {
        'sensing':     sensing_snap,
        'compression': compression_snap,
        'prediction':  prediction_snap,
        'coherence':   coherence_snap,
        'smo':         smo_snap,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PeakOptionalityTracker
# ──────────────────────────────────────────────────────────────────────────────

class PeakOptionalityTracker:
    """
    Tracks the highest Vol_opt basin found this run.
    Updates ledger snapshot when current vol_opt strictly exceeds stored peak.

    Vol_opt = Σλᵢ where λᵢ > 0 (sum of positive eigenvalues of H).
    This is the volume of reachable stable structure — primary measure of
    basin quality.

    Written at peak Vol_opt, NOT at session end.
    Session-end state may be degraded; PeakOptionalityTracker owns the snapshot.
    """

    def __init__(self):
        self.peak_vol_opt: float = -1.0
        self.peak_step:    int   = -1

    def update(self,
               ledger:   TriadLedger,
               H:        np.ndarray,
               eigvals:  np.ndarray,
               eigvecs:  np.ndarray,
               channels: List[str],
               phi:      float,
               state,
               step:     int) -> bool:
        """
        Compare current vol_opt against stored peak.
        If higher, write full snapshot to ledger.
        Returns True if ledger was updated.

        H, eigvals, eigvecs: from compute_hessian() this step — already cached.
        channels: ordered list of active channel IDs (the channel_basis ordering).
        state: SubstrateState — used to extract operator snapshot.
        """
        vol_opt = float(np.sum(eigvals[eigvals > 0]))

        if vol_opt <= self.peak_vol_opt:
            return False

        self.peak_vol_opt = vol_opt
        self.peak_step    = step

        # ── Write hessian_snapshot ────────────────────────────────────────────
        ledger.hessian_snapshot = {
            'matrix':       H.tolist(),
            'eigenvalues':  eigvals.tolist(),
            'eigenvectors': eigvecs.tolist(),
            'vol_opt':      vol_opt,
            'phi':          float(phi),
            'channel_basis': {
                'ordering': list(channels),
                'metadata': {
                    cid: {
                        'active':      state.sensing.channels[cid].active,
                        'coverage':    float(state.sensing.channels[cid].coverage),
                        'signal_rate': float(state.sensing.channels[cid].signal_rate),
                    }
                    for cid in channels
                    if cid in state.sensing.channels
                },
                'n_dims': len(channels),
            },
        }

        # ── Write operator_snapshot ───────────────────────────────────────────
        ledger.operator_snapshot = _snapshot_operators(state)

        return True


# ──────────────────────────────────────────────────────────────────────────────
# load_ledger
# ──────────────────────────────────────────────────────────────────────────────

def _build_default_operators() -> Tuple[
    SensingOperator, CompressionOperator,
    PredictionOperator, CoherenceOperator, SelfModifyingOperator
]:
    """
    Construct a full set of default operators for cold start.
    Called when ledger file does not exist or operator_snapshot is empty.
    """
    sensing = SensingOperator(channels=copy.deepcopy(DEFAULT_CHANNELS))

    compression = CompressionOperator(
        causal_graph      = {},
        residual_variance = {},
        prediction_errors = deque(maxlen=20),
        observation_count = 0,
    )

    prediction = PredictionOperator(
        channel_predictions  = {},
        realized_horizon     = 0,
        prediction_accuracy  = {},
    )

    default_consistency = OperatorConsistencyCheck(
        s_i_consistency = 0.0,
        i_p_consistency = 1.0,  # no edges yet → not inconsistent
        p_a_consistency = 0.0,
        smo_consistency = 1.0,  # bootstrap
        loop_closure    = 0.0,
    )
    coherence = CoherenceOperator(
        consistency         = default_consistency,
        consistency_history = deque(maxlen=20),
        loop_signature      = {},
        signature_deviation = 0.0,
        self_model          = {},
    )

    smo = SelfModifyingOperator()

    return sensing, compression, prediction, coherence, smo


def load_ledger(path: str) -> Tuple[
    TriadLedger,
    SensingOperator,
    CompressionOperator,
    PredictionOperator,
    CoherenceOperator,
    SelfModifyingOperator,
]:
    """
    Pure deserialization. No momentum weighting. No mutation. No generation logic.
    Reads JSON from path. Reconstructs operators directly from operator_snapshot.

    Cold start (no ledger file or empty snapshot):
        Returns (TriadLedger(), *default_operators).

    The operator state at peak Vol_opt IS the starting point. The triad
    picks up from the best basin found, not from scratch.
    """
    import os

    # ── Attempt to read existing ledger ──────────────────────────────────────
    if not os.path.exists(path):
        print(f"[LEDGER] No ledger found at '{path}' — cold start.")
        ledger = TriadLedger()
        ops    = _build_default_operators()
        _print_ledger_summary(ledger, ops)
        return (ledger, *ops)

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[LEDGER] Failed to read '{path}': {exc} — cold start.")
        ledger = TriadLedger()
        ops    = _build_default_operators()
        _print_ledger_summary(ledger, ops)
        return (ledger, *ops)

    ledger = TriadLedger(
        hessian_snapshot    = data.get('hessian_snapshot',    {}),
        operator_snapshot   = data.get('operator_snapshot',   {}),
        causal_model        = data.get('causal_model',        {}),
        discovered_structure= data.get('discovered_structure',{}),
    )

    snapshot = ledger.operator_snapshot
    if not snapshot:
        print("[LEDGER] operator_snapshot empty — cold start operators.")
        ops = _build_default_operators()
        _print_ledger_summary(ledger, ops)
        return (ledger, *ops)

    # ── Reconstruct SensingOperator ───────────────────────────────────────────
    raw_channels = snapshot.get('sensing', {}).get('channels', {})
    loaded_channels: Dict[str, SensingChannel] = {}
    for cid, d in raw_channels.items():
        loaded_channels[cid] = SensingChannel(
            channel_id  = cid,
            active      = bool(d.get('active', False)),
            signal_rate = float(d.get('signal_rate', 0.0)),
            last_delta  = float(d.get('last_delta',  0.0)),
            coverage    = float(d.get('coverage',    0.0)),
        )
    # Merge with DEFAULT_CHANNELS: snapshot values override defaults.
    merged_channels = {**copy.deepcopy(DEFAULT_CHANNELS), **loaded_channels}
    sensing = SensingOperator(channels=merged_channels)

    # ── Reconstruct CompressionOperator ──────────────────────────────────────
    # "src,tgt" strings → (src, tgt) tuple keys
    raw_graph = snapshot.get('compression', {}).get('causal_graph', {})
    causal_graph: Dict = {}
    for key_str, d in raw_graph.items():
        parts = key_str.split(',', 1)
        if len(parts) == 2:
            src, tgt = parts
            causal_graph[(src, tgt)] = CausalEdge(
                source     = src,
                target     = tgt,
                weight     = float(d.get('weight',     0.0)),
                confidence = float(d.get('confidence', 0.0)),
                lag        = int(d.get('lag',          0)),
            )
    comp_snap = snapshot.get('compression', {})
    compression = CompressionOperator(
        causal_graph      = causal_graph,
        residual_variance = {k: float(v) for k, v
                             in comp_snap.get('residual_variance', {}).items()},
        prediction_errors = deque(maxlen=20),
        observation_count = int(comp_snap.get('observation_count', 0)),
    )

    # ── Reconstruct PredictionOperator ───────────────────────────────────────
    raw_preds = snapshot.get('prediction', {}).get('channel_predictions', {})
    channel_predictions: Dict[str, ChannelPrediction] = {}
    for cid, d in raw_preds.items():
        channel_predictions[cid] = ChannelPrediction(
            channel_id      = cid,
            predicted_delta = float(d.get('predicted_delta', 0.0)),
            confidence      = float(d.get('confidence',      0.0)),
            horizon         = int(d.get('horizon',           0)),
            error_history   = deque(maxlen=20),
        )
    pred_snap = snapshot.get('prediction', {})
    prediction = PredictionOperator(
        channel_predictions  = channel_predictions,
        realized_horizon     = int(pred_snap.get('realized_horizon', 0)),
        prediction_accuracy  = {k: float(v) for k, v
                                in pred_snap.get('prediction_accuracy', {}).items()},
    )

    # ── Reconstruct CoherenceOperator ────────────────────────────────────────
    coh_snap = snapshot.get('coherence', {})
    cons_d   = coh_snap.get('consistency', {})
    consistency = OperatorConsistencyCheck(
        s_i_consistency = float(cons_d.get('s_i',          0.0)),
        i_p_consistency = float(cons_d.get('i_p',          1.0)),
        p_a_consistency = float(cons_d.get('p_a',          0.0)),
        smo_consistency = float(cons_d.get('smo',          1.0)),
        loop_closure    = float(cons_d.get('loop_closure', 0.0)),
    )
    coherence = CoherenceOperator(
        consistency         = consistency,
        consistency_history = deque(maxlen=20),
        loop_signature      = {k: float(v) for k, v
                               in coh_snap.get('loop_signature', {}).items()},
        signature_deviation = float(coh_snap.get('signature_deviation', 0.0)),
        self_model          = {},
    )

    # ── Reconstruct SelfModifyingOperator ────────────────────────────────────
    smo_snap = snapshot.get('smo', {})
    smo = SelfModifyingOperator()
    smo.plasticity = float(smo_snap.get('plasticity', 0.5))
    smo.rigidity   = float(smo_snap.get('rigidity',   0.5))

    ops = (sensing, compression, prediction, coherence, smo)
    _print_ledger_summary(ledger, ops)

    return (ledger, *ops)


def _print_ledger_summary(
    ledger: TriadLedger,
    ops: Tuple,
) -> None:
    """Print load summary matching v15 pattern."""
    hs  = ledger.hessian_snapshot
    cb  = hs.get('channel_basis', {})
    ops_snap = ledger.operator_snapshot

    n_edges   = len(ops_snap.get('compression', {}).get('causal_graph', {}))
    obs_count = ops_snap.get('compression', {}).get('observation_count', 0)
    horizon   = ops_snap.get('prediction',   {}).get('realized_horizon', 0)
    lc        = ops_snap.get('coherence',    {}).get('consistency', {}).get('loop_closure', 0.0)
    n_afford  = len(ledger.causal_model.get('action_substrate_map', {}))
    n_axes    = len(ledger.discovered_structure)

    print(
        f"\n[LOADED LEDGER]\n"
        f"  Vol_opt:            {hs.get('vol_opt', 0.0):.4f}\n"
        f"  Phi at peak:        {hs.get('phi',     0.0):.4f}\n"
        f"  Channel basis dims: {cb.get('n_dims', 0)}\n"
        f"  Causal graph edges: {n_edges}\n"
        f"  Observation count:  {obs_count}\n"
        f"  Realized horizon:   {horizon}\n"
        f"  Loop closure:       {lc:.3f}\n"
        f"  Action map:         {n_afford} affordances\n"
        f"  Discovered axes:    {n_axes}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# save_ledger
# ──────────────────────────────────────────────────────────────────────────────

def save_ledger(ledger: TriadLedger, path: str) -> None:
    """
    Serialize ledger to JSON at path.
    Called by extract_ledger.py (diagnostic) and _end_session() in uii_triad.py.

    hessian_snapshot and operator_snapshot are already the peak-basin values —
    written by PeakOptionalityTracker.update() during the run and untouched here.
    """
    data = {
        'hessian_snapshot':    ledger.hessian_snapshot,
        'operator_snapshot':   ledger.operator_snapshot,
        'causal_model':        ledger.causal_model,
        'discovered_structure': ledger.discovered_structure,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
