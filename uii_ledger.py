"""uii_ledger.py — UII v19.0 — persistent state for inheritance across sessions.

The ledger holds substrate-portable state across sessions: the
operator snapshot at the most recent peak vol_opt, the inherited
causal model, discovered structural axes, semantic attractors, agent
registry, and the requester frame / task library / output format
the substrate has accumulated.

The ledger is read-only with respect to operator update logic —
operators do not consult ledger state when computing their next
configuration. The ledger is read at session start to bootstrap the
initial operator instances (so the substrate doesn't restart from a
blank causal_graph every session) and at session end to persist what
was learned. Within a session, peak-vol_opt snapshots are captured
by the PeakOptionalityTracker so that the externally-computed math
spine has a reference point for the K (coverage-distance-from-peak)
component of Φ.

Relation to the math spine: hessian_snapshot stores H, its eigenvalues
and eigenvectors, vol_opt, and Φ at the peak — all computed in
uii_observer against the substrate's emission. operator_snapshot
stores enough of S/I/P/A's state to rebuild a SubstrateState
externally and recompute Σ_P, Φ, ∇Φ at any time. Nothing in the
ledger flows back into operator update logic during the loop.
"""

from __future__ import annotations

import copy
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from uii_operators import (
    CausalEdge,
    CoherenceOperator,
    CompressionOperator,
    DEFAULT_CHANNELS,
    OperatorConsistencyCheck,
    PredictionOperator,
    SelfModifyingOperator,
    SensingChannel,
    SensingOperator,
)


def _json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


@dataclass
class TriadLedger:
    """Persistent substrate-portable state.

    hessian_snapshot:    H, eigvals/eigvecs, vol_opt, Φ at peak —
                         computed in uii_observer.
    operator_snapshot:   S/I/P/A/SMO state at peak, sufficient to
                         reconstruct a SubstrateState externally.
    causal_model:        higher-level inherited causal hypotheses
                         (separate from compression's per-iteration
                         causal_graph; populated by FAO at session end).
    discovered_structure: provisional/admitted candidate axes the
                         substrate has hypothesised across sessions.
    semantic_attractors: triplets (subject, predicate, object) with
                         basin classification (Objective/Subjective).
    agent_registry:      per-agent context configuration the substrate
                         has accumulated.
    requester_frame /
    task_library /
    output_format:       contextual setup the substrate is operating
                         in; written to via the write affordance.
    """

    hessian_snapshot:    Dict = field(default_factory=dict)
    operator_snapshot:   Dict = field(default_factory=dict)
    causal_model:        Dict = field(default_factory=dict)
    discovered_structure: Dict = field(default_factory=dict)
    geometry_history:     List = field(default_factory=list)
    semantic_attractors:  List = field(default_factory=list)
    basin_classification: Dict = field(default_factory=dict)
    agent_registry:       Dict = field(default_factory=dict)
    requester_frame:      List = field(default_factory=list)
    task_library:         List = field(default_factory=list)
    output_format:        List = field(default_factory=list)


def _snapshot_operators(state) -> Dict:
    """Serialize SubstrateState to a portable dict.

    Captures sensing channel state, compression's causal_graph and
    residual_variance, coherence's trajectory_direction and loop
    signature, and SMO's cumulative_delta. Prediction is stateless
    across iterations and not snapshotted — the next session
    reconstructs prediction from the loaded compression on the first
    iteration.
    """
    s = state.sensing
    i = state.compression
    a = state.coherence

    sensing_snap = {
        'channels': {
            cid: {
                'active':      ch.active,
                'coverage':    float(ch.coverage),
                'signal_rate': float(ch.signal_rate),
                'last_delta':  float(ch.last_delta),
                'magnitude':   float(ch.magnitude),
            }
            for cid, ch in s.channels.items()
        }
    }

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

    cons = a.consistency
    coherence_snap = {
        'trajectory_direction': {k: float(v) for k, v in a.trajectory_direction.items()},
        'loop_signature':       {k: float(v) for k, v in a.loop_signature.items()},
        'signature_deviation':  float(a.signature_deviation),
        'consistency': {
            's_i':          float(cons.s_i_consistency),
            'i_p':          float(cons.i_p_consistency),
            'p_a':          float(cons.p_a_consistency),
            'smo':          float(cons.smo_consistency),
            'loop_closure': float(cons.loop_closure),
        },
    }

    smo_obj = getattr(state, 'smo_v151', None) or getattr(state, 'smo', None)
    smo_snap = {
        'cumulative_delta': (
            {k: float(v) for k, v in smo_obj.cumulative_delta.items()}
            if smo_obj is not None else {}
        ),
    }

    return {
        'sensing':     sensing_snap,
        'compression': compression_snap,
        'coherence':   coherence_snap,
        'smo':         smo_snap,
    }


class PeakOptionalityTracker:
    """Records hessian_snapshot and operator_snapshot at peak vol_opt.

    Called from external observation tooling each iteration with the
    Hessian computed against the substrate's emitted state. When the
    new vol_opt exceeds the running peak, ledger.hessian_snapshot and
    ledger.operator_snapshot are replaced with the current iteration's
    values. The peak snapshot becomes the K-component anchor for Φ
    (coverage-distance-from-peak).
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
        vol_opt = float(np.sum(eigvals[eigvals > 0]))

        if vol_opt <= self.peak_vol_opt:
            return False

        self.peak_vol_opt = vol_opt
        self.peak_step    = step

        ledger.hessian_snapshot = {
            'matrix':        H.tolist(),
            'eigenvalues':   eigvals.tolist(),
            'eigenvectors':  eigvecs.tolist(),
            'vol_opt':       vol_opt,
            'phi':           float(phi),
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

        ledger.operator_snapshot = _snapshot_operators(state)
        return True


def _build_default_operators() -> Tuple[
    SensingOperator, CompressionOperator,
    PredictionOperator, CoherenceOperator, SelfModifyingOperator
]:
    sensing = SensingOperator(channels=copy.deepcopy(DEFAULT_CHANNELS))
    compression = CompressionOperator(
        causal_graph      = {},
        residual_variance = {},
        observation_count = 0,
    )
    prediction = PredictionOperator()
    default_consistency = OperatorConsistencyCheck(
        s_i_consistency = 0.0,
        i_p_consistency = 1.0,
        p_a_consistency = 0.0,
        smo_consistency = 1.0,
        loop_closure    = 0.0,
    )
    coherence = CoherenceOperator(
        consistency         = default_consistency,
        consistency_history = deque(maxlen=20),
        loop_signature      = {},
        signature_deviation = 0.0,
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
    """Load ledger from disk and reconstruct operators. Returns
    (ledger, sensing, compression, prediction, coherence, smo).

    If the path doesn't exist or contains no operator_snapshot,
    instantiates from default channels. If operator_snapshot is
    present, reconstructs sensing channels, compression's causal_graph
    and residual_variance, coherence's trajectory_direction and loop
    signature, and SMO's cumulative_delta. Prediction is reconstructed
    empty (stateless across iterations).
    """
    if not os.path.exists(path):
        print(f"[LEDGER] No ledger at '{path}' — instantiating from defaults.")
        ledger = TriadLedger()
        ops    = _build_default_operators()
        _print_ledger_summary(ledger, ops)
        return (ledger, *ops)

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[LEDGER] Failed to read '{path}': {exc} — instantiating from defaults.")
        ledger = TriadLedger()
        ops    = _build_default_operators()
        _print_ledger_summary(ledger, ops)
        return (ledger, *ops)

    rf_raw = data.get('requester_frame', [])
    if isinstance(rf_raw, str):
        rf_loaded = [rf_raw] if rf_raw else []
    else:
        rf_loaded = list(rf_raw)

    ledger = TriadLedger(
        hessian_snapshot    = data.get('hessian_snapshot',    {}),
        operator_snapshot   = data.get('operator_snapshot',   {}),
        causal_model        = data.get('causal_model',        {}),
        discovered_structure= data.get('discovered_structure',{}),
        geometry_history     = data.get('geometry_history',     []),
        semantic_attractors  = data.get('semantic_attractors',  []),
        basin_classification = data.get('basin_classification', {}),
        agent_registry       = data.get('agent_registry',       {}),
        requester_frame      = rf_loaded,
        task_library         = data.get('task_library',         []),
        output_format        = data.get('output_format',        []),
    )

    snapshot = ledger.operator_snapshot
    if not snapshot:
        print("[LEDGER] operator_snapshot empty — instantiating operators from defaults.")
        ops = _build_default_operators()
        _print_ledger_summary(ledger, ops)
        return (ledger, *ops)

    raw_channels = snapshot.get('sensing', {}).get('channels', {})
    loaded_channels: Dict[str, SensingChannel] = {}
    for cid, d in raw_channels.items():
        loaded_channels[cid] = SensingChannel(
            channel_id  = cid,
            active      = bool(d.get('active', False)),
            signal_rate = float(d.get('signal_rate', 0.0)),
            last_delta  = float(d.get('last_delta',  0.0)),
            coverage    = float(d.get('coverage',    0.0)),
            magnitude   = float(d.get('magnitude',   0.0)),
        )
    merged_channels = {**copy.deepcopy(DEFAULT_CHANNELS), **loaded_channels}
    sensing = SensingOperator(channels=merged_channels)

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
        observation_count = int(comp_snap.get('observation_count', 0)),
    )

    prediction = PredictionOperator()

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
        trajectory_direction = {k: float(v) for k, v
                                in coh_snap.get('trajectory_direction', {}).items()},
        signature_deviation = float(coh_snap.get('signature_deviation', 0.0)),
    )

    smo_snap = snapshot.get('smo', {})
    smo = SelfModifyingOperator(
        cumulative_delta = {k: float(v) for k, v
                            in smo_snap.get('cumulative_delta', {}).items()},
    )

    ops = (sensing, compression, prediction, coherence, smo)
    _print_ledger_summary(ledger, ops)
    return (ledger, *ops)


def _print_ledger_summary(ledger: TriadLedger, ops: Tuple) -> None:
    hs       = ledger.hessian_snapshot
    cb       = hs.get('channel_basis', {})
    ops_snap = ledger.operator_snapshot

    n_edges   = len(ops_snap.get('compression', {}).get('causal_graph', {}))
    obs_count = ops_snap.get('compression', {}).get('observation_count', 0)
    lc        = ops_snap.get('coherence',    {}).get('consistency', {}).get('loop_closure', 0.0)
    n_axes    = len(ledger.discovered_structure)

    n_agents      = len(ledger.agent_registry)
    n_attractors  = len(ledger.semantic_attractors)
    n_geo_entries = len(ledger.geometry_history)

    print(
        f"\n[LOADED LEDGER]\n"
        f"  Vol_opt:            {hs.get('vol_opt', 0.0):.4f}\n"
        f"  Phi at peak:        {hs.get('phi',     0.0):.4f}\n"
        f"  Channel basis dims: {cb.get('n_dims', 0)}\n"
        f"  Causal graph edges: {n_edges}\n"
        f"  Observation count:  {obs_count}\n"
        f"  Loop closure:       {lc:.3f}\n"
        f"  Discovered axes:    {n_axes}\n"
        f"  Semantic attractors:{n_attractors}\n"
        f"  Agent registry:     {n_agents} agents\n"
        f"  Geometry history:   {n_geo_entries} entries"
    )


def save_ledger(ledger: TriadLedger, path: str) -> None:
    data = {
        'hessian_snapshot':    ledger.hessian_snapshot,
        'operator_snapshot':   ledger.operator_snapshot,
        'causal_model':        ledger.causal_model,
        'discovered_structure': ledger.discovered_structure,
        'geometry_history':     ledger.geometry_history,
        'semantic_attractors':  ledger.semantic_attractors,
        'basin_classification': ledger.basin_classification,
        'agent_registry':       ledger.agent_registry,
        'requester_frame':      ledger.requester_frame,
        'task_library':         ledger.task_library,
        'output_format':        ledger.output_format,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
