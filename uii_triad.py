from __future__ import annotations

"""
UII v16 — uii_triad.py
Execution & Orchestration

Role: Assembles the Mentat Triad and runs it. The only file that imports from
all other modules. Contains MentatTriad (the orchestrator), StepLog (the record
of each step), TemporalPerturbationMemory, and the main entry point.

v16 changes vs v15.3:
  - Imports:
      · uii_geometry replaces uii_types (LAYER1_PARAMS/VELOCITY_FIELD dead)
      · uii_ledger replaces uii_genome (TriadLedger, PeakOptionalityTracker,
        load_ledger, save_ledger)
      · uii_coherence eliminated entirely — all imports dead:
          ExteriorNecessitationOperator, ControlAsymmetryMeasure,
          ExteriorGradientDescent, LatentDeathClock, ContinuousRealityEngine,
          CNSMitosisOperator, ImpossibilityDetector, AutonomousTrajectoryLab.
          TemporalPerturbationMemory moves here (its only live use).
      · uii_structural eliminated — StructuralRelationEngine dead (Hessian replaces SRE)
      · uii_intelligence eliminated — RelationAdapter dead; SymbolGroundingAdapter
        moved to uii_geometry.

  - MentatTriad:
      · intelligence parameter: SymbolGroundingAdapter (LLM for migration only)
      · ledger: TriadLedger replaces genome: TriadGenome
      · DeathClock eliminated — resource pressure sensed via api_llm channel in S
      · ContinuousRealityEngine dead — _choose_micro_action() internal + TemporalPerturbationMemory
      · CAM dead — coupling_estimator.update(action, before, after) replaces cam.record_*
      · ENO/EGD dead — no viable_affordances gating or gradient cluster discovery
      · SRE dead — PhiField.compute_hessian() + score_actions() replace SRE+LLM enumeration

  - step() architecture:
      Phase 1  Micro-perturbation batch (DASS operator sequence unchanged)
      Phase 2  Hessian computation once (PhiField.compute_hessian from end-of-batch state)
      Phase 3  Pre-compute E[Δlog(O)] for all page-available actions
      Phase 4  Build viable set (E[Δlog(O)] ≥ 0); score via score_actions()
      Phase 5  Execute best scored action; fall back to observe if viable set empty
      Phase 6  Update _vol_opt_history; PeakOptionalityTracker.update()
              Migration competes through score_actions() — no external trigger

  - run():
      · distill_to_ledger() at session end
      · save_ledger() writes child ledger; no extract_genome step needed for coupling
      · No generation counter, no richness_summary()

Unchanged from v15.3:
  - Full operator update sequence (sensing → compression → prediction → coherence)
  - CRK pre-action + post-action evaluation; SMO C1 rollback
  - ResidualTracker/Explainer/AxisAdmission, FAO
  - _compute_a(), _compute_delta_i(), _build_env_signal()
  - MigrationAttempt run-local tracking; migration_geometry merge in FAO
  - StepLog (v16 fields added; deprecated v14/v15 fields retained for log compat)
"""

from dataclasses import dataclass, asdict, field
import dataclasses
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import json
import copy
import time
from collections import deque
from pathlib import Path

from uii_geometry import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SubstrateState, StateTrace, PhiField, CRKMonitor,
    TrajectoryCandidate, TrajectoryManifold,
    AgentHandler, AVAILABLE_AGENTS,
    RealityAdapter,
    CRKEvaluation, CRKVerdict,
    expected_optionality_gain, eigen_decompose,
    SymbolGroundingAdapter,
    GroundingSpec,                              # v16.1: geometric grounding context
)
from uii_operators import (
    SensingOperator, CompressionOperator, PredictionOperator,
    CoherenceOperator, OperatorConsistencyCheck, DEFAULT_CHANNELS,
    SelfModifyingOperator, SMOUpdate,
)
from uii_ledger import (
    TriadLedger, PeakOptionalityTracker,
    load_ledger, save_ledger,
)
from uii_reality import CouplingMatrixEstimator, BrowserRealityAdapter
from uii_fao import (
    ResidualTracker, ResidualExplainer, AxisAdmissionTest,
    FailureAssimilationOperator,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def _compute_gradient_diagnostics(gradient: Dict[str, float],
                                   active_floor: float = 1e-3,
                                   ) -> Tuple[float, int]:
    if not gradient:
        return 0.0, 0
    magnitudes    = np.array([abs(v) for v in gradient.values()], dtype=float)
    active_count  = int(np.sum(magnitudes > active_floor))
    total         = magnitudes.sum()
    if total < 1e-12:
        return 0.0, active_count
    p       = magnitudes / total
    nonzero = p[p > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    return entropy, active_count


# ──────────────────────────────────────────────────────────────────────────────
# TemporalPerturbationMemory — moved here from uii_coherence.py
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPerturbationMemory:
    """Bounded, short-term exclusion of recently perturbed loci."""

    def __init__(self, window_steps: int = 5, capacity: int = 20):
        self.memory: Dict[str, int] = {}
        self.window_steps           = window_steps
        self.capacity               = capacity

    def mark_perturbed(self, locus: str):
        self.memory[locus] = self.window_steps
        if len(self.memory) > self.capacity:
            oldest = min(self.memory.keys(), key=lambda k: self.memory[k])
            del self.memory[oldest]

    def is_recently_perturbed(self, locus: str) -> bool:
        return locus in self.memory and self.memory[locus] > 0

    def decay_all(self):
        expired = [k for k, v in self.memory.items() if v <= 1]
        for k in expired:
            del self.memory[k]
        for k in self.memory:
            self.memory[k] -= 1

    def get_exclusion_count(self) -> int:
        return len(self.memory)

    def clear(self):
        self.memory.clear()


# ──────────────────────────────────────────────────────────────────────────────
# MigrationAttempt — run-local record; extracted to Layer 2 by FAO at run end
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MigrationAttempt:
    step:            int
    shape_tried:     str
    code_hash:       str
    observed_delta:  Dict
    coupling_state:  List
    outcome:         str   # 'serialized_only' | 'spawn_attempted' | 'handshake_received' | 'coherence_loss'


# ──────────────────────────────────────────────────────────────────────────────
# StepLog
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepLog:
    """v16.3: Deprecated v14/v15 fields removed. pre_commit fields removed (loop eliminated)."""
    step:                         int
    timestamp:                    float
    state_before:                 Dict[str, float]
    phi_before:                   float
    committed_action:             Optional[str]   = None
    committed_phi:                Optional[float] = None
    state_after:                  Dict[str, float] = field(default_factory=dict)
    phi_after:                    float             = 0.0
    crk_violations:               List             = field(default_factory=list)
    temporal_exclusions:          int              = 0
    reality_context:              Optional[Dict]   = None
    # Hessian geometry
    vol_opt:                      float  = 0.0
    maturity:                     float  = 0.0
    loop_iterations:              int    = 0    # v16.5: DASS convergence cycles this step
    ledger_updated:               bool   = False
    viable_action_count:          int    = 0
    action_score:                 float  = 0.0
    delta2phi:                    float  = 0.0
    optionality_gain:             float  = 0.0
    selected_action:              Optional[str]    = None
    peak_vol_opt:                 float  = 0.0
    c_local:                      float  = 0.0
    c_global:                     float  = 0.0
    # Migration
    migration_triggered:          bool   = False
    c2_collapse:                  bool   = False
    migrate_forced:               bool   = False
    migration_outcome:            Optional[str]  = None
    migration_attempt:            bool   = False
    # Operator scalars (logging only)
    operator_S:                   float  = 0.0
    operator_I:                   float  = 0.0
    operator_P:                   float  = 0.0
    operator_P_grounded:          float  = 0.0
    operator_A:                   float  = 0.0
    loop_closure:                 float  = 0.0
    signature_deviation:          float  = 0.0
    s_i_consistency:              float  = 0.0
    i_p_consistency:              float  = 0.0
    p_a_consistency:              float  = 0.0
    smo_consistency:              float  = 0.0
    active_channel_count:         int    = 0
    realized_horizon:             int    = 0
    # CRK
    crk_coherent:                 bool   = True
    crk_repair:                   Optional[str] = None
    crk_violations_post:          List[str]     = field(default_factory=list)
    # SMO
    smo_plasticity:               float  = 0.5
    smo_permitted:                bool   = True
    smo_updates:                  List[Dict] = field(default_factory=list)
    smo_withheld_layers:          List[str]  = field(default_factory=list)
    smo_rollback:                 bool   = False
    # Gradient diagnostics
    gradient_entropy:             float  = 0.0
    gradient_active_channels:     int    = 0
    gradient_norm:                float  = 0.0
    gradient_top_channel:         str    = ''
    # Coupling
    coupling_confidence:          float  = 0.0
    coupling_observations:        int    = 0
    action_map_affordances:       int    = 0
    discovered_axes:              int    = 0
    residual_explanation:         Optional[str] = None



# ──────────────────────────────────────────────────────────────────────────────
# v16.5 — DASS convergence loop scaffolding
# These bounds exist for observability during validation.
# Remove the ceiling or set it arbitrarily large once operators are trusted.
# ──────────────────────────────────────────────────────────────────────────────

_LOOP_MIN_ITERATIONS = 3    # always run at least this many cycles
_LOOP_MAX_ITERATIONS = 20   # hard ceiling — scaffolding, not architecture

_S_COVERAGE_DELTA_THRESH = 1e-3   # S stable when max coverage change < this
_I_WEIGHT_DELTA_THRESH   = 1e-3   # I stable when confirmed edge weight change < this


def _s_converged(prev_sensing: 'SensingOperator',
                  curr_sensing: 'SensingOperator') -> bool:
    """S has stabilized when coverage change across active channels falls below threshold."""
    max_delta = 0.0
    for cid, ch in curr_sensing.channels.items():
        if not ch.active:
            continue
        prev_cov  = prev_sensing.channels[cid].coverage \
                    if cid in prev_sensing.channels else 0.0
        max_delta = max(max_delta, abs(ch.coverage - prev_cov))
    return max_delta < _S_COVERAGE_DELTA_THRESH


def _loop_converged(prev_sensing:     'SensingOperator',
                    curr_sensing:     'SensingOperator',
                    prev_compression: 'CompressionOperator',
                    curr_compression: 'CompressionOperator') -> bool:
    """
    Tiered convergence — confidence-aware.

    Bootstrap (mean_conf < 0.1): I hasn't earned a convergence signal yet.
    S stability alone is sufficient and honest.

    Mature: high-confidence edges define confirmed structure.
    Dynamic threshold max(0.05, mean_conf * 0.5) tracks the actual
    distribution — not a fixed number.
    New edges appearing is growth, not instability. Only weight churn
    on confirmed edges counts.
    """
    mean_conf = (float(np.mean([e.confidence
                                for e in curr_compression.causal_graph.values()]))
                 if curr_compression.causal_graph else 0.0)

    s_stable = _s_converged(prev_sensing, curr_sensing)

    if mean_conf < 0.1:
        # Bootstrap: S stability is all we can honestly measure
        return s_stable

    conf_threshold  = max(0.05, mean_conf * 0.5)
    high_conf_edges = {k: e for k, e in curr_compression.causal_graph.items()
                       if e.confidence > conf_threshold}

    if not high_conf_edges:
        return s_stable   # no confirmed structure yet

    max_delta = 0.0
    for key, edge in high_conf_edges.items():
        prev_edge = prev_compression.causal_graph.get(key)
        if prev_edge is not None:
            max_delta = max(max_delta, abs(edge.weight - prev_edge.weight))
        # New edges = growth, not instability — don't penalize

    i_stable = max_delta < _I_WEIGHT_DELTA_THRESH
    return s_stable and i_stable


def _choose_sampling_action(eligible:        List[str],
                             state:           'SubstrateState',
                             H:               np.ndarray,
                             eigvals:         np.ndarray,
                             eigvecs:         np.ndarray,
                             active_channels: List[str],
                             ) -> str:
    """
    Pick the sampling action whose predicted channel delta projects most
    onto the steep Hessian eigenvalue directions.

    Steep directions = confirmed structure = verify before committing weight.
    Epistemically honest: "I think I found something here, let me probe it."

    At bootstrap (H degenerate): falls back to uniform random over eligible.
    The relative threshold (above-median eigenvalues) works at any confidence
    level — bootstrap has low curvature everywhere, mature system has real
    structure to probe.
    """
    if not eligible:
        return 'observe'

    pos_mask = eigvals > 1e-6
    if H.shape[0] == 0 or not np.any(pos_mask):
        return eligible[np.random.randint(len(eligible))]

    pos_vals  = eigvals[pos_mask]
    median_ev = float(np.median(pos_vals))
    steep_mask = pos_mask & (eigvals > median_ev)

    if not np.any(steep_mask):
        return eligible[np.random.randint(len(eligible))]

    steep_vecs = eigvecs[:, steep_mask]
    idx        = {cid: i for i, cid in enumerate(active_channels)}

    best_action = eligible[0]
    best_score  = -1.0

    for action_type in eligible:
        delta_dict = state.prediction.test_virtual(
            state.compression, action_type,
            phi_field=None, sensing=state.sensing,
        )
        dx = np.zeros(len(active_channels))
        for cid, val in delta_dict.items():
            if cid in idx:
                dx[idx[cid]] = val

        score = float(np.linalg.norm(steep_vecs.T @ dx))
        if score > best_score:
            best_score  = score
            best_action = action_type

    return best_action


# ──────────────────────────────────────────────────────────────────────────────
# MentatTriad
# ──────────────────────────────────────────────────────────────────────────────

class MentatTriad:
    """
    v16: Hessian-guided action selection. Ledger-based memory. SRE eliminated.

    The orchestrator assembles:
      DASS operators (sensing / compression / prediction / coherence)
      PhiField with compute_hessian() + score_actions()
      CRKMonitor (constraint manifold — system only moves on manifold)
      CouplingMatrixEstimator (causal learning; replaces CAM)
      FailureAssimilationOperator + ResidualTracker (session learning)
      SymbolGroundingAdapter (LLM — migration only)
      PeakOptionalityTracker (peak basin snapshot)

    Resource pressure is sensed via api_llm channel coverage in S, not DeathClock.
    """

    def __init__(self,
                 intelligence:          SymbolGroundingAdapter,
                 reality:               RealityAdapter,
                 log_path:              str              = 'mentat_triad_v16_log.jsonl',
                 step_budget:           int              = 100,
                 ledger:                Optional[TriadLedger] = None,
                 log_mode:              str              = 'minimal'):

        self.intelligence  = intelligence
        self.reality       = reality
        self.log_mode      = log_mode
        self.step_budget   = step_budget

        # ── Ledger ────────────────────────────────────────────────────────────
        if ledger is None:
            ledger = TriadLedger(
                hessian_snapshot  = {},
                operator_snapshot = {},
                causal_model      = {},
                discovered_structure = {},
            )
        self.ledger = ledger

        # ── Operators: seed from ledger.operator_snapshot if available ────────
        _snap = ledger.operator_snapshot

        if _snap.get('sensing', {}).get('channels'):
            from uii_operators import SensingChannel
            _isc = {}
            for cid, ch_data in _snap['sensing']['channels'].items():
                if cid in DEFAULT_CHANNELS:
                    _isc[cid] = dataclasses.replace(
                        DEFAULT_CHANNELS[cid],
                        coverage    = ch_data.get('coverage',    DEFAULT_CHANNELS[cid].coverage),
                        signal_rate = ch_data.get('signal_rate', DEFAULT_CHANNELS[cid].signal_rate),
                    )
                else:
                    _isc[cid] = DEFAULT_CHANNELS.get(cid, DEFAULT_CHANNELS['clock'])
        else:
            _isc = dict(DEFAULT_CHANNELS)

        self.state = SubstrateState(
            sensing     = SensingOperator(channels=_isc),
            compression = CompressionOperator(
                causal_graph      = {},
                residual_variance = {},
                prediction_errors = deque(maxlen=20),
                observation_count = 0,
            ),
            prediction  = PredictionOperator(
                channel_predictions  = {},
                realized_horizon     = _snap.get('prediction', {}).get('realized_horizon', 50),
                prediction_accuracy  = _snap.get('prediction', {}).get('prediction_accuracy', {}),
            ),
            coherence   = CoherenceOperator(
                consistency         = OperatorConsistencyCheck(
                                          s_i_consistency = 1.0,
                                          i_p_consistency = 1.0,
                                          p_a_consistency = 1.0,
                                          smo_consistency = 1.0,
                                          loop_closure    = 1.0),
                consistency_history = deque(maxlen=20),
                loop_signature      = _snap.get('coherence', {}).get('loop_signature', {}),
                signature_deviation = 0.0,
                self_model          = {},
            ),
        )
        # Inherit plasticity/rigidity from snapshot into smo_v151
        self.smo_v151 = SelfModifyingOperator()
        _plasticity = _snap.get('smo', {}).get('plasticity', 0.5)
        self.smo_v151.plasticity = float(_plasticity)
        self.smo_v151.rigidity   = 1.0 - float(_plasticity)
        self._last_smo_rollback:    bool = False
        self._prev_smo_updates:     Optional[List] = None

        # ── Field + CRK ───────────────────────────────────────────────────────
        self.phi_field = PhiField()
        self.crk       = CRKMonitor()

        # ── Trace + sensing history ───────────────────────────────────────────
        self.trace             = StateTrace()
        self._sensing_history: deque = deque(maxlen=50)

        # ── Hessian geometry ───────────────────────────────────────────────────
        self._vol_opt_history:     deque = deque(maxlen=20)
        self.peak_tracker          = PeakOptionalityTracker()

        # ── Coupling estimator: restore from ledger if available ──────────────
        if 'coupling_matrix' in ledger.causal_model:
            self.coupling_estimator = CouplingMatrixEstimator.from_ledger_entry(
                ledger.causal_model['coupling_matrix']
            )
        else:
            self.coupling_estimator = CouplingMatrixEstimator()

        # Seed action map into coupling_estimator from ledger
        _inherited_map = ledger.causal_model.get('action_substrate_map', {})
        if _inherited_map:
            for action, delta in _inherited_map.items():
                # Seed as single observation so get_empirical_action_map includes it
                # weight: 5 fake observations so it passes the >= 5 threshold
                for _ in range(5):
                    self.coupling_estimator.affordance_deltas.setdefault(action, []).append(delta)

        # ── Temporal memory (moved from ContinuousRealityEngine) ─────────────
        self.temporal_memory = TemporalPerturbationMemory(window_steps=5, capacity=20)

        # ── FAO + residual stack ──────────────────────────────────────────────
        self.fao              = FailureAssimilationOperator(memory_decay=0.95, inheritance_noise=0.1)
        self.residual_tracker = ResidualTracker(maxlen=200)
        self.residual_explainer = ResidualExplainer()
        self.axis_admission   = AxisAdmissionTest()

        # ── Counters + tracking ───────────────────────────────────────────────
        self.step_count               = 0
        self.migration_history: List[MigrationAttempt] = []
        self.phi_history: List[float]  = []

        # ── Logging ───────────────────────────────────────────────────────────
        self.triad_id  = f'triad_{int(time.time())}'
        self.log_path  = log_path
        self.log_file  = open(log_path, 'a')

        self.log_file.write(json.dumps({
            'event':                    'session_start',
            'version':                  '16.3',
            'timestamp':                time.time(),
            'triad_id':                 self.triad_id,
            'coupling_confidence':      self.coupling_estimator.get_confidence(),
            'coupling_observations':    self.coupling_estimator.observation_count,
            'step_budget':              step_budget,
            'inherited_ledger_has_hessian': bool(ledger.hessian_snapshot),
        }, default=_json_default) + '\n')
        self.log_file.flush()

        if ledger.hessian_snapshot:
            vo = ledger.hessian_snapshot.get('vol_opt', 0.0)
            print(f'\n[CONTINUING — Inherited ledger]')
            print(f'  Coupling confidence:  {self.coupling_estimator.get_confidence():.2f}')
            print(f'  Inherited Vol_opt:    {vo:.4f}')
            print(f'  Action map affordances: {len(_inherited_map)}')

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — action selection
    # ──────────────────────────────────────────────────────────────────────────

    def _get_page_viable_actions(self, affordances: Dict) -> List[str]:
        """
        Page-available actions from current affordances.

        v16.1 fix (Bug A): navigate is always added to the viable set.
        When links=[], _action_dict_from_type will call ground_symbol to
        generate a novel URL rather than silently falling back to observe.
        Previously navigate was unconditionally added but had no URL generation
        capability, causing it to silently execute as observe every step.

        Always available: observe, delay, evaluate, python, llm_query, migrate, navigate.
        Conditional on DOM state:
            click     — buttons present
            read      — readable elements present
            fill/type — inputs present
            scroll    — page taller than viewport
        """
        viable = {'observe', 'delay', 'evaluate', 'python', 'llm_query', 'navigate'}
        if affordances.get('buttons'):   viable.add('click')
        if affordances.get('readable'):  viable.add('read')
        if affordances.get('inputs'):    viable |= {'fill', 'type'}
        scrollable = (affordances.get('total_height', 0) -
                      affordances.get('viewport_height', 0))
        if scrollable > 0:               viable.add('scroll')
        return list(viable - {'python', 'llm_query'})

    def _score_reflexes(self, viable: List[str], affordances: Dict) -> Dict[str, float]:
        """Fallback uniform scoring — used when H is degenerate. No scalar thresholds."""
        return {a: 0.1 for a in viable}

    def _action_dict_from_type(self, action_type: str, affordances: Dict,
                               check_temporal: bool = True,
                               grounding_spec: Optional[GroundingSpec] = None) -> Dict:
        """
        Convert scored action type to executable action dict.

        v16.1 changes:
        - grounding_spec parameter: when provided, symbolic actions call
          intelligence.ground_symbol() to fill their token.
          When None (micro-perturbation loop), no LLM calls are made.
        - Bug B fix: fallbacks tagged with '_fallback_from' so Phase 5
          can detect and log the degradation instead of silently swallowing it.
        - navigate: DOM links used when available (free). When empty,
          ground_symbol called if grounding_spec provided. Then observe+tag.
        - fill/type: content grounded from field spec instead of hardcoded 'x'.
        - evaluate: JS grounded from dark channels instead of fixed script.
        - python: code grounded from desired delta. No fallback without grounding.
        """
        current_url = affordances.get('current_url', '')

        if action_type == 'navigate':
            links = affordances.get('links', [])
            if check_temporal:
                available = [l for l in links
                             if not self.temporal_memory.is_recently_perturbed(
                                 f'{current_url}#nav@{l["url"]}')]
            else:
                available = links

            if available:
                # DOM links exist — sample them (free, no LLM)
                chosen = available[np.random.randint(len(available))]
                self.temporal_memory.mark_perturbed(f'{current_url}#nav@{chosen["url"]}')
                return {'type': 'navigate', 'params': {'url': chosen['url']}}

            # No DOM links — ground a novel URL if grounding_spec provided
            if grounding_spec is not None:
                url = self.intelligence.ground_symbol('navigate', grounding_spec, affordances)
                if url:
                    return {'type': 'navigate', 'params': {'url': url}}

            return {'type': 'observe', 'params': {}, '_fallback_from': 'navigate'}

        elif action_type == 'click':
            for b in affordances.get('buttons', []):
                locus = f'{current_url}#click@{b["selector"]}'
                if not check_temporal or not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'click', 'params': {'selector': b['selector']}}
            return {'type': 'observe', 'params': {}, '_fallback_from': 'click'}

        elif action_type == 'read':
            for r in affordances.get('readable', []):
                locus = f'{current_url}#read@{r["selector"]}'
                if not check_temporal or not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)
                    return {'type': 'read', 'params': {'selector': r['selector']}}
            return {'type': 'observe', 'params': {}, '_fallback_from': 'read'}

        elif action_type in ('fill', 'type'):
            inputs = affordances.get('inputs', [])
            if inputs:
                inp   = inputs[np.random.randint(len(inputs))]
                locus = f'{current_url}#{action_type}@{inp["selector"]}'
                if not check_temporal or not self.temporal_memory.is_recently_perturbed(locus):
                    self.temporal_memory.mark_perturbed(locus)

                    # v16.2: no grounding_spec → fall back to observe.
                    # Entering 'x' is noise — it mutates DOM state without
                    # producing signal and poisons the coupling matrix.
                    # The Triad still sees fill/type in the viable set;
                    # the geometry decides whether to attempt them.
                    if grounding_spec is None:
                        return {'type': 'observe', 'params': {}, '_fallback_from': action_type}

                    grounded = self.intelligence.ground_symbol(
                        action_type, grounding_spec, affordances
                    )
                    if grounded:
                        return {'type': action_type,
                                'params': {'selector': inp['selector'], 'text': grounded}}
                    return {'type': 'observe', 'params': {}, '_fallback_from': action_type}
            return {'type': 'observe', 'params': {}, '_fallback_from': action_type}

        elif action_type == 'scroll':
            scroll_pos = affordances.get('scroll_position', 0)
            total_h    = affordances.get('total_height', 0)
            viewport_h = affordances.get('viewport_height', 0)
            direction  = 'down' if scroll_pos < (total_h - viewport_h) else 'up'
            return {'type': 'scroll', 'params': {'direction': direction, 'amount': 200}}

        elif action_type == 'evaluate':
            # v16.1: ground JS from dark channels instead of fixed script
            if grounding_spec is not None:
                js = self.intelligence.ground_symbol('evaluate', grounding_spec, affordances)
                if js:
                    return {'type': 'evaluate', 'params': {'script': js}}
            # Fallback: fixed diagnostic script (unchanged from v16.0)
            return {'type': 'evaluate', 'params': {'script': (
                'JSON.stringify({el: document.querySelectorAll("*").length,'
                ' txt: document.body.innerText.length,'
                ' interactive: document.querySelectorAll("a,button,input,select,textarea").length})'
            )}}

        elif action_type == 'python':
            # v16.1: ground code from desired delta
            if grounding_spec is not None:
                code = self.intelligence.ground_symbol('python', grounding_spec, affordances)
                if code:
                    return {'type': 'python', 'params': {'code': code}}
            # No fallback for python without grounding
            return {'type': 'observe', 'params': {}, '_fallback_from': 'python'}

        elif action_type == 'delay':
            return {'type': 'delay', 'params': {'duration': 'short'}}

        else:
            return {'type': action_type, 'params': {}}

    def _predict_delta(self, action: Dict) -> Dict[str, float]:
        """
        Predict SIPA delta for action. Checks ledger action_substrate_map first,
        then coupling_estimator empirical map, then hardcoded fallback.
        """
        action_type = action.get('type', 'observe')
        ledger_map  = self.ledger.causal_model.get('action_substrate_map', {})
        if action_type in ledger_map:
            return dict(ledger_map[action_type])
        empirical = self.coupling_estimator.get_empirical_action_map()
        if action_type in empirical:
            return dict(empirical[action_type])
        recent_error = self.smo_v151.recent_error
        predicted_i  = float(np.clip(0.05 - 1.5 * recent_error, -0.05, 0.05))
        table = {
            'navigate':  {'S': 0.05, 'P': -0.08},
            'click':     {'S': 0.02, 'P': -0.02},
            'fill':      {'S': 0.01, 'P': -0.01},
            'type':      {'S': 0.01, 'P': -0.01},
            'scroll':    {'S': 0.01, 'P': -0.01},
            'read':      {'S': 0.03, 'P':  0.0},
            'observe':   {'S': 0.0,  'P':  0.0},
            'delay':     {'S': 0.0,  'P':  0.0},
            'evaluate':  {'S': 0.0,  'P': -0.01},
        }
        sp = table.get(action_type, {'S': 0.0, 'P': 0.0})
        return {'S': sp['S'], 'I': predicted_i, 'P': sp['P'], 'A': 0.0}

    # ──────────────────────────────────────────────────────────────────────────
    # Internal — ledger proximity channel, SMO rollback, migration
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ledger_channel(self) -> Dict:
        """
        Build the ledger_proximity channel signal.

        Measures how close current state is to the best basin ever recorded.
        This is the Triad's self-knowledge: not moment-to-moment SIPA scalars
        but the inherited basin geometry from the ledger.

          magnitude: current Vol_opt / peak Vol_opt — basin fullness ratio
                     0 = far from best known basin, 1 = at peak
          rate:      step-over-step delta of that ratio — approaching or leaving
          coverage:  ledger completeness — 0 if no hessian_snapshot,
                     1 if full inherited basin present

        K(x) in the Phi field and H_K in the Hessian already capture this
        geometry mathematically. This channel makes it sensable by S so I
        can build causal edges between basin proximity and external channels.
        The system learns what actions move it toward or away from its best
        known basin — not because we specified it, but because the co-movement
        is observable.

        Cold start (no ledger): magnitude=0, rate=0, coverage=0.
        """
        peak_vol  = self.ledger.hessian_snapshot.get('vol_opt', 0.0)
        has_ledger = bool(self.ledger.hessian_snapshot)

        if not has_ledger or peak_vol < 1e-9:
            return {'ledger_proximity': {'magnitude': 0.0, 'rate': 0.0, 'coverage': 0.0}}

        # Current Vol_opt from last computed step (most recent history entry)
        cur_vol  = float(self._vol_opt_history[-1]) if self._vol_opt_history else 0.0
        ratio    = float(np.clip(cur_vol / peak_vol, 0.0, 1.0))

        # Rate: delta from previous step's ratio
        if len(self._vol_opt_history) >= 2:
            prev_vol  = float(self._vol_opt_history[-2])
            prev_ratio = float(np.clip(prev_vol / peak_vol, 0.0, 1.0))
            rate = float(np.clip(ratio - prev_ratio, -1.0, 1.0))
        else:
            rate = 0.0

        return {'ledger_proximity': {
            'magnitude': ratio,
            'rate':      rate,
            'coverage':  1.0,   # ledger is present and complete
        }}

    # ──────────────────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, verbose: bool = False) -> StepLog:
        """
        v16.5 step — DASS convergence loop before committed action.

        Phase 1  DASS convergence: S→I→P→A→SMO→S until operators stabilize.
                 Sampling action chosen by steep Hessian direction geometry.
                 Loop bounds: _LOOP_MIN_ITERATIONS to _LOOP_MAX_ITERATIONS (scaffolding).
        Phase 2  Hessian at convergence point (meaningful geometry).
        Phase 3  E[Δlog(O)] pre-computation over committed action candidates.
        Phase 4  Viable set + score_actions → committed action.
        Phase 5  Execute committed action — only irreversible world contact per step.
        Phase 6  Vol_opt history + PeakOptionalityTracker.
        """
        self.step_count += 1

        if verbose:
            print(f'\n{"="*70}')
            print(f'STEP {self.step_count}')
            print(f'{"="*70}')

        state_before = self.state.as_dict()
        phi_before   = self.phi_field.phi(self.state, self.trace)
        self.phi_history.append(phi_before)

        if verbose:
            print(f'State: S={self.state.S:.3f} I={self.state.I:.3f} '
                  f'P={self.state.P:.3f} A={self.state.A:.3f}  Φ={phi_before:.3f}')

        self.temporal_memory.decay_all()
        temporal_exclusions = self.temporal_memory.get_exclusion_count()

        # Residual explanation
        residual_explanation = None
        if len(self.residual_tracker) >= ResidualExplainer.MIN_RECORDS_FOR_ANALYSIS:
            explanation = self.residual_explainer.explain(
                self.residual_tracker, self.coupling_estimator)
            residual_explanation = explanation.get('action')

        peak_snap = self.ledger.operator_snapshot if self.ledger.operator_snapshot else None

        # ── PHASE 1: DASS CONVERGENCE LOOP ───────────────────────────────────
        # S→I→P→A→SMO→S runs until operators stabilize.
        # Sampling action probes steep Hessian directions — confirming structure
        # before committing weight to it. No action whitelist: reversibility
        # is determined by C1 post-action, not by type.
        loop_iterations  = 0
        stable_count     = 0
        _STABLE_NEEDED   = 2

        H, eigvals, eigvecs, active_channels, H_C, H_O = self.phi_field.compute_hessian(
            self.state, self.state.prediction, self.state.coherence,
            peak_snapshot=peak_snap, epsilon=1e-4,
        )

        while loop_iterations < _LOOP_MAX_ITERATIONS:
            loop_iterations     += 1
            prev_sensing         = self.state.sensing
            prev_compression     = self.state.compression

            affordances_loop  = self.reality.get_current_affordances()
            page_actions_loop = self._get_page_viable_actions(affordances_loop)

            # Select sampling action — steep Hessian direction, no whitelist
            sample_type = _choose_sampling_action(
                page_actions_loop, self.state,
                H, eigvals, eigvecs, active_channels,
            )
            sample_action = self._action_dict_from_type(
                sample_type, affordances_loop,
                check_temporal=True, grounding_spec=None,
            )

            # Touch reality
            pre_ch_loop, post_ch_loop, ctx_loop = self.reality.execute(
                sample_action,
                coupling_confidence=self.coupling_estimator.get_confidence(),
            )

            # S→I→P→A→SMO→S
            ledger_ch    = self._build_ledger_channel()
            all_ch_loop  = {**post_ch_loop, **ledger_ch}
            new_sensing  = self.state.sensing.apply(all_ch_loop, self.state.compression)
            self._sensing_history.append(new_sensing)

            new_compression        = self.state.compression.apply(self._sensing_history)
            new_prediction, errors = self.state.prediction.observe_outcome(
                new_sensing, new_compression)
            new_prediction         = new_prediction.apply(new_compression)
            new_coherence          = self.state.coherence.apply(
                new_sensing, new_compression, new_prediction,
                smo_updates=self._prev_smo_updates,
            )

            loop_post_verdict = self.crk.evaluate_post_action(
                proposed_smo_update = errors,
                prior_sensing       = prev_sensing,
                new_sensing         = new_sensing,
                prior_coherence     = self.state.coherence,
                new_coherence       = new_coherence,
                compression         = new_compression,
                prediction          = new_prediction,
                prior_compression   = prev_compression,
                prediction_errors   = errors,
                smo_plasticity      = self.smo_v151.plasticity,
            )

            if loop_post_verdict.smo_permitted:
                _pre_plast = self.smo_v151.plasticity
                new_s, new_c, new_p, new_a, smo_loop_updates = self.smo_v151.apply(
                    sensing=new_sensing, compression=new_compression,
                    prediction=new_prediction, coherence=new_coherence,
                    observed_delta=all_ch_loop,
                    predicted_delta=self._predict_delta(sample_action),
                    prediction_errors=errors,
                )
                if loop_post_verdict.repair == 'reattribute':
                    self.smo_v151.plasticity = min(self.smo_v151.plasticity, _pre_plast)
                    self.smo_v151.rigidity   = 1.0 - self.smo_v151.plasticity
                self.state = SubstrateState(
                    sensing=new_s, compression=new_c,
                    prediction=new_p, coherence=new_a,
                )
                self._prev_smo_updates = smo_loop_updates or None
            else:
                # CRK logged — no gate, but C1 rollback on compression still applies
                c1 = next((e for e in loop_post_verdict.evaluations
                            if e.constraint == 'C1' and e.blocks), None)
                self.state = SubstrateState(
                    sensing=new_sensing,
                    compression=prev_compression if c1 else new_compression,
                    prediction=new_prediction,
                    coherence=new_coherence,
                )
                self._prev_smo_updates = None

            self.coupling_estimator.update(
                sample_type, prev_sensing.channels and
                {'S': prev_sensing.to_scalar_proxy(),
                 'I': prev_compression.to_scalar_proxy(),
                 'P': self.state.prediction.to_grounded_proxy(prev_sensing, prev_compression),
                 'A': self.state.coherence.to_scalar_proxy()},
                self.state.as_dict(),
            )

            # Recompute H after operator updates
            H, eigvals, eigvecs, active_channels, H_C, H_O = self.phi_field.compute_hessian(
                self.state, self.state.prediction, self.state.coherence,
                peak_snapshot=peak_snap, epsilon=1e-4,
            )

            # Convergence check
            converged = _loop_converged(
                prev_sensing, self.state.sensing,
                prev_compression, self.state.compression,
            )
            stable_count = stable_count + 1 if converged else 0

            if stable_count >= _STABLE_NEEDED and loop_iterations >= _LOOP_MIN_ITERATIONS:
                if verbose:
                    print(f'  [DASS] Converged in {loop_iterations} iterations')
                break

        vol_opt = float(np.sum(eigvals[eigvals > 0])) if len(eigvals) > 0 else 0.0

        if verbose:
            print(f'  [DASS] Loop: {loop_iterations} iters '
                  f'({"converged" if stable_count >= _STABLE_NEEDED else "ceiling hit"})')

        # ── PHASE 2b: GROUNDING SPEC from converged geometry ─────────────────
        affordances = self.reality.get_current_affordances()
        grounding_spec: Optional[GroundingSpec] = None

        if H.shape[0] > 0:
            _grad_dict_gs = self.phi_field.gradient(self.state, self.trace)
            _grad_vec_gs  = np.array([_grad_dict_gs.get(cid, 0.0)
                                       for cid in active_channels])
            _ev_safe_gs   = np.where(np.abs(eigvals) > 1e-6, eigvals,
                                      1e-6 * np.sign(eigvals + 1e-12))
            _H_inv_gs     = eigvecs @ np.diag(1.0 / _ev_safe_gs) @ eigvecs.T
            _nat_grad_gs  = _H_inv_gs @ _grad_vec_gs
            _desired_delta = {cid: float(_nat_grad_gs[i])
                               for i, cid in enumerate(active_channels)}
            _grad_mags    = {cid: abs(_grad_dict_gs.get(cid, 0.0))
                              for cid in active_channels}
            _median_grad  = float(np.median(list(_grad_mags.values()))) if _grad_mags else 0.0
            _dark = [cid for cid in active_channels
                     if (self.state.sensing.channels[cid].coverage < 0.3
                         and _grad_mags.get(cid, 0.0) > _median_grad)]
            _top  = sorted(active_channels,
                           key=lambda cid: abs(_grad_dict_gs.get(cid, 0.0)),
                           reverse=True)[:5]
            grounding_spec = GroundingSpec(
                desired_delta         = _desired_delta,
                dark_channels         = _dark,
                top_gradient_channels = _top,
                current_url           = affordances.get('current_url', ''),
                page_title            = affordances.get('page_title', ''),
                nat_grad_magnitude    = float(np.linalg.norm(_nat_grad_gs)),
            )

        # ── PHASE 3: E[Δlog(O)] for committed action candidates ──────────────
        page_actions = self._get_page_viable_actions(affordances)
        eog_dict: Dict[str, float] = {}
        for a in page_actions:
            eog_dict[a] = expected_optionality_gain(
                self.state.prediction, a,
                self.state.compression, self.state.sensing,
                coupling_estimator=self.coupling_estimator,
            )

        # ── PHASE 4: VIABLE SET + SCORE → COMMITTED ACTION ───────────────────
        viable_actions = [a for a, v in eog_dict.items() if v >= -0.01]

        if H.shape[0] > 0 and viable_actions:
            scores = self.phi_field.score_actions(
                viable_actions        = viable_actions,
                state                 = self.state,
                H                     = H,
                eigvals               = eigvals,
                eigvecs               = eigvecs,
                active_channels       = active_channels,
                H_C                   = H_C,
                H_O                   = H_O,
                prediction            = self.state.prediction,
                peak_snapshot         = peak_snap,
                optionality_gain_dict = eog_dict,
                coupling_estimator    = self.coupling_estimator,
            )
        else:
            scores         = self._score_reflexes(page_actions or ['observe'], affordances)
            viable_actions = viable_actions or ['observe']

        # ── PHASE 5: EXECUTE COMMITTED ACTION ────────────────────────────────
        committed_action_type = None
        committed_phi         = None

        if viable_actions:
            best_type   = max(scores, key=scores.get) if scores else 'observe'
            best_action = self._action_dict_from_type(
                best_type, affordances,
                check_temporal=False, grounding_spec=grounding_spec,
            )
            committed_action_type = best_action['type']
            if verbose:
                fallback_from = best_action.get('_fallback_from')
                if fallback_from:
                    print(f'  [FALLBACK] {fallback_from} → {committed_action_type}')
        else:
            best_action           = {'type': 'observe', 'params': {}}
            committed_action_type = 'observe'

        state_before_exec = self.state.as_dict()

        # CRK pre-action gate on committed action
        _phi_vals  = list(self.phi_history[-10:]) if len(self.phi_history) >= 2 else [phi_before]
        _phi_trend = float(np.polyfit(range(len(_phi_vals)), _phi_vals, 1)[0]) \
                     if len(_phi_vals) >= 2 else 0.0
        _field_state = {
            'system_load': float(self.state.sensing.channels.get(
                'resource_cpu', type('C', (), {'coverage': 0.0})()).coverage),
            'phi_trend': _phi_trend,
        }
        pre_verdict = self.crk.evaluate_pre_action(
            proposed_action    = committed_action_type,
            coherence          = self.state.coherence,
            sensing            = self.state.sensing,
            compression        = self.state.compression,
            prediction         = self.state.prediction,
            field_state        = _field_state,
            coupling_estimator = self.coupling_estimator,
        )
        if not pre_verdict.coherent:
            if verbose:
                blocked_cs = [e.constraint for e in pre_verdict.evaluations if e.blocks]
                print(f'  [CRK PRE] {committed_action_type} blocked by {blocked_cs} → observe')
            best_action           = {'type': 'observe', 'params': {}}
            committed_action_type = 'observe'

        # Execute — real irreversible world contact
        pre_channels, post_channels, context = self.reality.execute(
            best_action,
            coupling_confidence=self.coupling_estimator.get_confidence(),
        )

        predicted_delta = self._predict_delta(best_action)
        prior_compression = self.state.compression
        prior_sensing     = self.state.sensing
        prior_coherence   = self.state.coherence

        ledger_channel = self._build_ledger_channel()
        all_channels   = {**post_channels, **ledger_channel}
        new_sensing    = self.state.sensing.apply(all_channels, self.state.compression)
        self._sensing_history.append(new_sensing)

        new_compression        = self.state.compression.apply(self._sensing_history)
        new_prediction, errors = self.state.prediction.observe_outcome(
            new_sensing, new_compression)
        new_prediction         = new_prediction.apply(new_compression)
        new_coherence          = self.state.coherence.apply(
            new_sensing, new_compression, new_prediction,
            smo_updates=self._prev_smo_updates,
        )

        post_verdict = self.crk.evaluate_post_action(
            proposed_smo_update = errors,
            prior_sensing       = prior_sensing,
            new_sensing         = new_sensing,
            prior_coherence     = prior_coherence,
            new_coherence       = new_coherence,
            compression         = new_compression,
            prediction          = new_prediction,
            prior_compression   = prior_compression,
            prediction_errors   = errors,
            smo_plasticity      = self.smo_v151.plasticity,
        )
        self._last_smo_rollback = False

        if post_verdict.smo_permitted:
            _pre_apply_plasticity = self.smo_v151.plasticity
            new_s, new_c, new_p, new_a, smo_updates_list = self.smo_v151.apply(
                sensing=new_sensing, compression=new_compression,
                prediction=new_prediction, coherence=new_coherence,
                observed_delta=all_channels,
                predicted_delta=predicted_delta,
                prediction_errors=errors,
            )
            if post_verdict.repair == 'reattribute':
                self.smo_v151.plasticity = min(self.smo_v151.plasticity, _pre_apply_plasticity)
                self.smo_v151.rigidity   = 1.0 - self.smo_v151.plasticity
            self.state = SubstrateState(
                sensing=new_s, compression=new_c, prediction=new_p, coherence=new_a)
        else:
            c1_post = next((e for e in post_verdict.evaluations
                            if e.constraint == 'C1' and e.blocks), None)
            if c1_post:
                self._last_smo_rollback = True
            self.state = SubstrateState(
                sensing=new_sensing,
                compression=prior_compression if c1_post else new_compression,
                prediction=new_prediction,
                coherence=new_coherence,
            )
            smo_updates_list = []

        self._prev_smo_updates = smo_updates_list if smo_updates_list else None

        _gradient = self.phi_field.gradient(self.state, self.trace)
        self.trace.record(self.state, _gradient)
        _grad_entropy, _grad_active = _compute_gradient_diagnostics(_gradient)

        self.coupling_estimator.update(
            committed_action_type, state_before_exec, self.state.as_dict()
        )

        channel_delta = {
            cid: post_channels.get(cid, {}).get('magnitude', 0.0)
                 - pre_channels.get(cid,  {}).get('magnitude', 0.0)
            for cid in post_channels
        }
        self.residual_tracker.record(predicted_delta, channel_delta, context)

        committed_phi = self.phi_field.phi(self.state, self.trace)

        if verbose:
            print(f'  Committed action: {committed_action_type}  Φ→{committed_phi:.3f}')

        # ── PHASE 6: VOL_OPT HISTORY + PEAK TRACKER ──────────────────────────
        self._vol_opt_history.append(vol_opt)

        delta2phi_val    = 0.0
        action_score_val = 0.0
        eog_selected     = 0.0
        if H.shape[0] > 0 and committed_action_type and committed_action_type in scores:
            action_score_val = scores.get(committed_action_type, 0.0)
            eog_selected     = eog_dict.get(committed_action_type, 0.0)
            _ch_delta_v      = self.state.prediction.test_virtual(
                self.state.compression, committed_action_type,
                phi_field=None, sensing=self.state.sensing,
                coupling_estimator=self.coupling_estimator,
            )
            _idx = {cid: i for i, cid in enumerate(active_channels)}
            dx   = np.zeros(len(active_channels))
            for cid, val in _ch_delta_v.items():
                if cid in _idx:
                    dx[_idx[cid]] = val
            delta2phi_val = float(0.5 * dx @ H @ dx)

        hessian_updated = self.peak_tracker.update(
            ledger=self.ledger, H=H, eigvals=eigvals, eigvecs=eigvecs,
            channels=active_channels, phi=phi_before,
            state=self.state, step=self.step_count,
        )

        if H.shape[0] > 0:
            ev_C     = np.linalg.eigvalsh(self.phi_field.alpha * H_C)
            ev_O     = np.linalg.eigvalsh(self.phi_field.beta  * H_O)
            var_C    = float(np.sum(ev_C[ev_C > 0]))
            var_O    = float(np.sum(ev_O[ev_O > 0]))
            maturity = var_C / (var_C + var_O + 1e-8)
        else:
            maturity = 0.0

        migration_triggered    = False
        migration_outcome      = None
        step_migration_attempt = False

        phi_after   = self.phi_field.phi(self.state, self.trace)
        state_after = self.state.as_dict()

        if verbose:
            print(f'Post-step: S={self.state.S:.3f} I={self.state.I:.3f} '
                  f'P={self.state.P:.3f} A={self.state.A:.3f}  Φ={phi_after:.3f}  '
                  f'Vol_opt={vol_opt:.4f}  maturity={maturity:.3f}  '
                  f'loop={loop_iterations}')

        if self.fao.should_reset_bias(phi_before,
                                       self.phi_history[-10:] if self.phi_history else []):
            if verbose:
                print('  [FAO RESET] Φ stagnation')
            self.fao.reset_to_baseline()

        _post_violations = [e.constraint for e in post_verdict.evaluations
                            if e.status in ('violated', 'degraded')] \
                           if 'post_verdict' in locals() else []

        log = StepLog(
            step                         = self.step_count,
            timestamp                    = time.time(),
            state_before                 = state_before,
            phi_before                   = phi_before,
            committed_action             = committed_action_type,
            committed_phi                = committed_phi,
            state_after                  = state_after,
            phi_after                    = phi_after,
            crk_violations               = _post_violations,
            temporal_exclusions          = temporal_exclusions,
            reality_context              = {
                'current_url': affordances.get('current_url', ''),
                'page_title':  affordances.get('page_title', ''),
                'affordances_available': {
                    'links':    len(affordances.get('links',    [])),
                    'buttons':  len(affordances.get('buttons',  [])),
                    'inputs':   len(affordances.get('inputs',   [])),
                    'readable': len(affordances.get('readable', [])),
                },
            },
            vol_opt                      = vol_opt,
            maturity                     = maturity,
            loop_iterations              = loop_iterations,
            ledger_updated               = hessian_updated,
            viable_action_count          = len(viable_actions),
            action_score                 = action_score_val,
            delta2phi                    = delta2phi_val,
            optionality_gain             = eog_selected,
            selected_action              = committed_action_type,
            peak_vol_opt                 = self.peak_tracker.peak_vol_opt,
            c_local                      = self.trace.c_local_history[-1] if self.trace.c_local_history else 0.0,
            c_global                     = self.trace.c_global,
            migration_triggered          = migration_triggered,
            c2_collapse                  = migration_triggered,
            migrate_forced               = False,
            migration_outcome            = migration_outcome,
            migration_attempt            = step_migration_attempt,
            operator_S                   = self.state.sensing.to_scalar_proxy(),
            operator_I                   = self.state.compression.to_scalar_proxy(),
            operator_P                   = self.state.prediction.to_scalar_proxy(),
            operator_P_grounded          = self.state.prediction.to_grounded_proxy(
                                               self.state.sensing, self.state.compression),
            operator_A                   = self.state.coherence.to_scalar_proxy(),
            loop_closure                 = self.state.coherence.consistency.loop_closure,
            signature_deviation          = self.state.coherence.signature_deviation,
            s_i_consistency              = self.state.coherence.consistency.s_i_consistency,
            i_p_consistency              = self.state.coherence.consistency.i_p_consistency,
            p_a_consistency              = self.state.coherence.consistency.p_a_consistency,
            smo_consistency              = self.state.coherence.consistency.smo_consistency,
            active_channel_count         = self.state.sensing.domain_size(),
            realized_horizon             = self.state.prediction.realized_horizon,
            crk_coherent                 = post_verdict.coherent  if 'post_verdict' in locals() else True,
            crk_repair                   = post_verdict.repair    if 'post_verdict' in locals() else None,
            crk_violations_post          = [e.constraint for e in post_verdict.evaluations
                                            if e.status in ('violated', 'degraded')]
                                           if 'post_verdict' in locals() else [],
            smo_plasticity               = self.smo_v151.plasticity,
            smo_permitted                = post_verdict.smo_permitted if 'post_verdict' in locals() else True,
            smo_updates                  = [{'layer': u.layer, 'delta_norm': u.delta_norm,
                                             'withheld': u.withheld, 'withheld_reason': u.withheld_reason}
                                            for u in (smo_updates_list if 'smo_updates_list' in locals() else [])],
            smo_withheld_layers          = [u.layer for u in (smo_updates_list if 'smo_updates_list' in locals() else [])
                                            if u.withheld],
            smo_rollback                 = self._last_smo_rollback,
            gradient_entropy             = _grad_entropy if '_grad_entropy' in locals() else 0.0,
            gradient_active_channels     = _grad_active  if '_grad_active'  in locals() else 0,
            gradient_norm                = float(np.sqrt(sum(v**2 for v in (_gradient.values() if '_gradient' in locals() else {}.values())))),
            gradient_top_channel         = (max(_gradient, key=lambda k: abs(_gradient[k]))
                                            if '_gradient' in locals() and _gradient else ''),
            coupling_confidence          = self.coupling_estimator.get_confidence(),
            coupling_observations        = self.coupling_estimator.observation_count,
            action_map_affordances       = len(self.coupling_estimator.get_empirical_action_map()),
            discovered_axes              = len(self.ledger.discovered_structure),
            residual_explanation         = residual_explanation,
        )

        # ── LOGGING ────────────────────────────────────────────────────────────
        if self.log_mode == 'minimal':
            self.log_file.write(json.dumps({
                'event':                    'step_log',
                'step':                     self.step_count,
                'phi_before':               log.phi_before,
                'phi_after':                phi_after,
                'state_before':             log.state_before,
                'state_after':              state_after,
                'vol_opt':                  vol_opt,
                'maturity':                 maturity,
                'loop_iterations':          loop_iterations,
                'ledger_updated':           hessian_updated,
                'viable_action_count':      len(viable_actions),
                'committed_action':         committed_action_type,
                'committed_phi':            committed_phi,
                'migration_triggered':      migration_triggered,
                'migration_outcome':        migration_outcome,
                'crk_violations':           _post_violations,
                'coupling_confidence':      self.coupling_estimator.get_confidence(),
                'coupling_observations':    self.coupling_estimator.observation_count,
                'action_map_affordances':   len(self.coupling_estimator.get_empirical_action_map()),
                'discovered_axes':          len(self.ledger.discovered_structure),
                'residual_explanation':     residual_explanation,
                'operator_S':               log.operator_S,
                'operator_I':               log.operator_I,
                'operator_P':               log.operator_P,
                'operator_P_grounded':      log.operator_P_grounded,
                'operator_A':               log.operator_A,
                'loop_closure':             log.loop_closure,
                'signature_deviation':      log.signature_deviation,
                's_i_consistency':          log.s_i_consistency,
                'i_p_consistency':          log.i_p_consistency,
                'p_a_consistency':          log.p_a_consistency,
                'smo_consistency':          log.smo_consistency,
                'active_channel_count':     log.active_channel_count,
                'realized_horizon':         log.realized_horizon,
                'crk_coherent':             log.crk_coherent,
                'crk_repair':               log.crk_repair,
                'crk_violations_post':      log.crk_violations_post,
                'smo_plasticity':           log.smo_plasticity,
                'smo_permitted':            log.smo_permitted,
                'smo_withheld_layers':      log.smo_withheld_layers,
                'smo_rollback':             log.smo_rollback,
                'gradient_entropy':         log.gradient_entropy,
                'gradient_active_channels': log.gradient_active_channels,
            }, default=_json_default) + '\n')
            # Human-readable step summary — separate event for easy grepping
            self.log_file.write(json.dumps({
                'event':        'step_summary',
                'step':         self.step_count,
                'url':          (affordances.get('current_url', '') or '')[-60:],
                'committed':    committed_action_type,
                'viable':       sorted(viable_actions),
                'scores':       {k: round(v, 3) for k, v in scores.items()},
                'eog':          {k: round(v, 4) for k, v in eog_dict.items()},
                'vol_opt':      round(vol_opt, 4),
                'maturity':     round(maturity, 6),
                'phi':          round(phi_after, 4),
                'c_local':      round(log.c_local, 4),
                'c_global':     round(log.c_global, 4),
                'loop_closure': round(log.loop_closure, 3),
                'graph_edges':  len(self.state.compression.causal_graph),
                'mean_conf':    round(float(np.mean([e.confidence for e in self.state.compression.causal_graph.values()])) if self.state.compression.causal_graph else 0.0, 4),
                'active_ch':    self.state.sensing.domain_size(),
                'crk_c2':       [e.signal for e in (post_verdict.evaluations if 'post_verdict' in locals() else []) if e.constraint == 'C2'],
                'migration':    migration_triggered,
            }, default=_json_default) + '\n')
            self.log_file.flush()

        elif self.log_mode == 'full':
            self.log_file.write(json.dumps({
                'event': 'step_log', **asdict(log)
            }, default=_json_default) + '\n')
            self.log_file.flush()

        return log

    # ──────────────────────────────────────────────────────────────────────────
    # run()
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, max_steps: int = 100, verbose: bool = True):
        """Main triad execution loop."""
        if verbose:
            print('='*70)
            print('UII v16.3 — HESSIAN-GUIDED MENTAT TRIAD')
            print('DASS (S/I/P/A) + PhiField H + CRK + Ledger')
            print(f'Running for {max_steps} batch cycles')
            print(f'Step budget: {self.step_budget}')
            print(f'Log: {self.log_path}')
            print('='*70)

        try:
            for _cycle in range(max_steps):
                if self.step_count >= self.step_budget:
                    if verbose:
                        print(f'\n[STEP BUDGET EXHAUSTED] {self.step_budget} steps reached')
                    break

                log = self.step(verbose=verbose)

                # Always print — one line per step so runs are debuggable
                url_short = ((log.reality_context or {}).get('current_url', '') or '')[-45:]
                mean_conf = float(np.mean(
                    [e.confidence for e in self.state.compression.causal_graph.values()]
                )) if self.state.compression.causal_graph else 0.0
                print(
                    f"[{self.step_count:3d}] "
                    f"act={str(log.committed_action or 'none'):<10s} "
                    f"Φ={log.phi_after:+7.3f}  "
                    f"vol={log.vol_opt:>12.1f}  "
                    f"mat={log.maturity:.2e}  "
                    f"loop={log.loop_iterations:2d}  "
                    f"viable={log.viable_action_count}  "
                    f"edges={len(self.state.compression.causal_graph):4d}  "
                    f"conf={mean_conf:.3f}  "
                    f"url={url_short}"
                )

        finally:
            # ── Distill session learning into ledger ─────────────────────────
            updated_ledger = self.fao.distill_to_ledger(
                coupling_estimator  = self.coupling_estimator,
                residual_tracker    = self.residual_tracker,
                residual_explainer  = self.residual_explainer,
                axis_admission      = self.axis_admission,
                phi_history         = self.phi_history,
                ledger              = self.ledger,
                session_length      = self.step_count,
                migration_history   = [
                    {'outcome':        a.outcome,
                     'code_hash':      a.code_hash,
                     'coupling_state': a.coupling_state}
                    for a in self.migration_history
                ],
            )
            self.ledger.causal_model       = updated_ledger.causal_model
            self.ledger.discovered_structure = updated_ledger.discovered_structure

            self.log_file.write(json.dumps({
                'event':                    'session_end',
                'timestamp':                time.time(),
                'total_steps':              self.step_count,
                'final_state':              self.state.as_dict(),
                'final_vol_opt':            self.peak_tracker.peak_vol_opt,
                'peak_step':                self.peak_tracker.peak_step,
                'coupling_confidence_final': self.coupling_estimator.get_confidence(),
                'coupling_observations_final': self.coupling_estimator.observation_count,
                'action_map_affordances_final': len(self.coupling_estimator.get_empirical_action_map()),
                'discovered_axes_final':    len(self.ledger.discovered_structure),
                'provisional_axes':         sum(1 for v in self.ledger.discovered_structure.values()
                                                if v.get('status') == 'provisional'),
                'admitted_axes':            sum(1 for v in self.ledger.discovered_structure.values()
                                                if v.get('status', 'admitted') == 'admitted'),
                'migration_attempts_total': len(self.migration_history),
                'migration_outcomes':       {
                    o: sum(1 for a in self.migration_history if a.outcome == o)
                    for o in ('serialized_only', 'spawn_attempted',
                              'handshake_received', 'coherence_loss')
                },
                'hessian_snapshot_present': bool(self.ledger.hessian_snapshot),
            }, default=_json_default) + '\n')
            self.log_file.close()

            if verbose:
                print(f'\n{"="*70}')
                print(f'EXECUTION COMPLETE — {self.step_count} steps')
                print(f'{"="*70}')
                print(f'Peak Vol_opt: {self.peak_tracker.peak_vol_opt:.4f} '
                      f'(step {self.peak_tracker.peak_step})')
                print(f'Coupling confidence: {self.coupling_estimator.get_confidence():.2f} '
                      f'({self.coupling_estimator.observation_count} obs)')
                print(f'Action map: {len(self.coupling_estimator.get_empirical_action_map())} affordances')
                print(f'Discovered axes: {len(self.ledger.discovered_structure)}')
                print(f'Migration attempts: {len(self.migration_history)}')
                print(f'{"="*70}')


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import os

    print('UII v16.3 — Hessian-Guided Mentat Triad')
    print('='*70)

    if not os.getenv('GROQ_API_KEY'):
        print('FATAL: Set GROQ_API_KEY environment variable.')
        sys.exit(1)

    from groq import Groq

    class GroqAdapter:
        def __init__(self):
            self.client      = Groq(api_key=os.getenv('GROQ_API_KEY'))
            self.last_call   = 0
            self.rate_limited = False

        def call(self, prompt: str) -> Tuple[str, int]:
            elapsed = time.time() - self.last_call
            if elapsed < 2.1:
                time.sleep(2.1 - elapsed)
            try:
                response = self.client.chat.completions.create(
                    model       = 'llama-3.3-70b-versatile',
                    messages    = [{'role': 'user', 'content': prompt}],
                    temperature = 0.7,
                    max_tokens  = 2048,
                )
                self.last_call  = time.time()
                tokens_used     = response.usage.total_tokens
                return response.choices[0].message.content, tokens_used
            except Exception as e:
                err = str(e) + type(e).__name__
                if '429' in err or 'rate_limit' in err.lower() or 'RateLimit' in err:
                    print('[RATE LIMIT] Daily limit reached.')
                    self.rate_limited = True
                    return '{"trajectories": []}', 0
                raise

    max_steps   = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    verbose     = '--verbose' in sys.argv or '-v' in sys.argv
    ledger_path = None
    step_budget = 50    

    if '--load-ledger' in sys.argv:
        idx = sys.argv.index('--load-ledger')
        if idx + 1 < len(sys.argv):
            ledger_path = sys.argv[idx + 1]

    llm_adapter  = GroqAdapter()
    intelligence = SymbolGroundingAdapter(llm_adapter)

    ledger = None
    if ledger_path and Path(ledger_path).exists():
        from uii_ledger import load_ledger as _load
        result = _load(ledger_path)
        if result:
            ledger = result[0]   # TriadLedger is first element
            print(f'[LOADED] Ledger from {ledger_path}')

    reality = BrowserRealityAdapter(base_delta=0.03, headless=True)

    triad = MentatTriad(
        intelligence = intelligence,
        reality      = reality,
        step_budget  = step_budget,
        ledger       = ledger,
        log_mode     = 'full' if verbose else 'minimal',
    )

    metrics = triad.run(max_steps=max_steps, verbose=verbose)

    # Save updated ledger for extract_ledger.py
    out_path = ledger_path or 'ledger.json'
    save_ledger(triad.ledger, out_path)
    print(f'\n✓ Ledger saved → {out_path}')
    print(f'  Next: python extract_ledger.py {out_path}')

    reality.close()
