from __future__ import annotations

"""uii_triad.py — UII v19.0 — execution and orchestration.

The only module that imports from all other UII modules. Assembles
the substrate and runs the closure loop S → I → P → A → SMO → S.
Defines MentatTriad — the orchestrator that wires the four operators
and SMO together with reality adapters and the ledger.

iterate() runs one step. The body is the strict-I/O wiring: each
operator's apply takes a single positional argument that is the
prior operator's output. env_signal accumulates each iteration from
probes (reality adapters), self introspection (Python clock/GC/
process metrics), affordance availability, and SMO's relation_signals
stashed from the previous iteration. Sensing reads env_signal once
and produces the perceptual surface; the rest of the loop follows.

The commit gate is coherence.commit_decision — a structural answer
produced inside the substrate's closure loop. No external EOG
ranking, no argmax over a quality measure, no ranking-stability
heuristic. When commit_decision is non-None the orchestrator
dispatches the action through the reality adapter; otherwise it
holds and re-audits next iteration.

Logging — the substrate emits one JSONL record per iteration on
stdout:

    {"iter": N, "t": ..., "commit": <action_dict|null>,
     "delta_f_rel": {channel_id: delta, ...}}

commit is the action_dict-as-emitted (the cause the substrate
pushed into the world) or null when no commit fired. delta_f_rel
is the closure residual SMO produced this iteration, per channel,
non-zero entries only — the substrate's perception of motion. This
is the substrate's structural emission. Nothing else. Math-spine
readings (Φ, ∇Φ, vol_opt, C_local, C_global), session banners,
periodic summaries — all live in the interface layer, which
consumes this stream alongside a query path into substrate state.
uii_observer remains the math-spine module; it is not imported
here.

stderr carries execution faults (Python tracebacks, adapter
errors). The JSONL stream on stdout is never interrupted.

Relation to the math spine: per source-of-truth §11, instantiated
systems are in-protocol by construction. The loop either runs or it
doesn't; whether the running loop is in the regime is what an
external observer reads from the emitted state-trajectory.
"""

from typing import Dict, List, Tuple, Optional, Set
import dataclasses
import numpy as np
import json
import sys
import time
from collections import deque
from pathlib import Path

from uii_geometry import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SubstrateState, StateTrace,
    AgentHandler, AVAILABLE_AGENTS,
    RealityAdapter,
    GroundingSpec,
    SELF_AFFORDANCES,
    RELATION_AFFORDANCES,
)
from uii_operators import (
    SensingOperator, CompressionOperator, PredictionOperator,
    CoherenceOperator, OperatorConsistencyCheck, DEFAULT_CHANNELS,
    SelfModifyingOperator, SensingChannel,
)
from uii_ledger import (
    TriadLedger,
    load_ledger, save_ledger,
)
from uii_reality import (
    BrowserRealityAdapter,
    AgentRealityAdapter,
    UserRealityAdapter,
    CompositeRealityAdapter,
    AttractorMonitor,
)


def _json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


class TemporalPerturbationMemory:
    """Bounded buffer of recent commits for trajectory analysis."""

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


class MentatTriad:
    """Substrate orchestrator.

    Owns the per-iteration SubstrateState (S, I, P, A operator instances)
    and the persistent SMO instance. Wires reality adapters (probe,
    execute, get_current_affordances) and the ledger (substrate-portable
    state across sessions).

    iterate() runs one step of the closure loop and emits one JSONL
    record on stdout: {iter, t, commit, delta_f_rel}. run() iterates
    until KeyboardInterrupt; respond_to_query lets external code feed
    an agent response back into the substrate's perceptual surface.

    No internal scoring, no parallel curator, no math-spine readings
    inside the substrate — those live in the interface layer, which
    consumes the JSONL stream alongside a query path into substrate
    state.
    """

    def __init__(self,
                 llm_client:            object,
                 reality:               RealityAdapter,
                 ledger:                Optional[TriadLedger] = None):
        self.reality = reality

        if ledger is None:
            ledger = TriadLedger(
                hessian_snapshot     = {},
                operator_snapshot    = {},
                causal_model         = {},
                discovered_structure = {},
            )
        self.ledger = ledger

        _snap = ledger.operator_snapshot

        if _snap.get('sensing', {}).get('channels'):
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
                observation_count = 0,
            ),
            prediction  = PredictionOperator(),
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
            ),
        )
        self.smo_v151 = SelfModifyingOperator(
            cumulative_delta = {
                k: float(v) for k, v in _snap.get('smo', {}).get('cumulative_delta', {}).items()
            },
        )

        self.attractor_monitor = AttractorMonitor()
        self.agent_reality     = AgentRealityAdapter(llm_client)
        self.user_reality      = UserRealityAdapter()
        self.composite_reality = CompositeRealityAdapter(
            browser           = self.reality,
            agent             = self.agent_reality,
            user              = self.user_reality,
            ledger            = self.ledger,
            get_state         = lambda: self.state,
            attractor_monitor = self.attractor_monitor,
        )

        # SMO's relation_signals from iteration N are stashed here and
        # merged into env_signal at the start of iteration N+1, entering
        # sensing through the same pathway as every other perturbation.
        self._pending_relation_signals: Dict = {}

        self._sensing_history: deque = deque(maxlen=50)

        self.temporal_memory = TemporalPerturbationMemory(window_steps=5, capacity=20)

        self._iteration_count: int = 0
        self._commit_count:    int = 0
        # _pending_action_attribution: the action just committed; consumed
        # by the next iteration's _build_affordance_channel to mark the
        # affordance as "just used" via a one-iteration magnitude/rate
        # spike. This is how committed actions enter compression's edge
        # discovery as first-class causal sources.
        self._pending_action_attribution: Optional[str] = None
        self._iteration_times: deque = deque(maxlen=20)

        self.triad_id = f'triad_{int(time.time())}'


    def _get_page_viable_actions(self, affordances: Dict) -> List[str]:
        viable = {'observe', 'delay', 'evaluate', 'navigate'}
        if affordances.get('buttons'):   viable.add('click')
        if affordances.get('readable'):  viable.add('read')
        if affordances.get('inputs'):    viable |= {'fill', 'type'}
        scrollable = (affordances.get('total_height', 0) -
                      affordances.get('viewport_height', 0))
        if scrollable > 0:               viable.add('scroll')
        return list(viable)


    def _select_library_entry(self, library: List, max_entries: int = 3) -> str:
        if not library:
            return ''
        recent = library[-max_entries:]
        return '\n'.join(str(e) for e in recent)

    def _format_attractors_for_prompt(self, basin_type: str, max_n: int = 5) -> str:
        candidates = [a for a in self.ledger.semantic_attractors
                      if a.get('basin_type') == basin_type]
        candidates.sort(key=lambda a: a.get('confidence', 0.0), reverse=True)
        candidates = candidates[:max_n]
        if not candidates:
            return '(none yet)'
        lines = []
        for a in candidates:
            sub = a.get('subject', '?')
            pre = a.get('predicate', '?')
            obj = a.get('object', '?')
            conf = a.get('confidence', 0.0)
            lines.append(f"  {sub} {pre} {obj} (conf={conf:.2f})")
        return '\n'.join(lines)

    def _extract_fill_text_from_history(self) -> Optional[str]:
        if not self.agent_reality._response_history:
            return None
        most_recent = self.agent_reality._response_history[-1]
        if not isinstance(most_recent, str) or not most_recent.strip():
            return None
        first_line = most_recent.strip().split('\n', 1)[0].strip()
        if 0 < len(first_line) <= 200:
            return first_line
        return None

    def _library_entry_from_history(self, library_label: str) -> Optional[str]:
        if not self.agent_reality._response_history:
            return None
        most_recent = self.agent_reality._response_history[-1]
        if not isinstance(most_recent, str) or not most_recent.strip():
            return None
        text = most_recent.strip()
        if len(text) > 500:
            text = text[:500] + '…'
        return text

    def _select_write_target(self) -> str:
        candidates = [
            ('task_library',    len(self.ledger.task_library)),
            ('requester_frame', len(self.ledger.requester_frame)),
            ('output_format',   len(self.ledger.output_format)),
        ]
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _action_dict_from_type(self, action_type: str, affordances: Dict,
                               check_temporal: bool = True,
                               grounding_spec: Optional[GroundingSpec] = None) -> Dict:
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
                chosen = available[np.random.randint(len(available))]
                self.temporal_memory.mark_perturbed(f'{current_url}#nav@{chosen["url"]}')
                return {'type': 'navigate', 'params': {'url': chosen['url']}}

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

                    fill_text = self._extract_fill_text_from_history()
                    if fill_text:
                        return {'type': action_type,
                                'params': {'selector': inp['selector'], 'text': fill_text}}
                    return {'type': 'observe', 'params': {}, '_fallback_from': action_type}
            return {'type': 'observe', 'params': {}, '_fallback_from': action_type}

        elif action_type == 'scroll':
            scroll_pos = affordances.get('scroll_position', 0)
            total_h    = affordances.get('total_height', 0)
            viewport_h = affordances.get('viewport_height', 0)
            direction  = 'down' if scroll_pos < (total_h - viewport_h) else 'up'
            return {'type': 'scroll', 'params': {'direction': direction, 'amount': 200}}

        elif action_type == 'evaluate':
            return {'type': 'evaluate', 'params': {'script': (
                'JSON.stringify({el: document.querySelectorAll("*").length,'
                ' txt: document.body.innerText.length,'
                ' interactive: document.querySelectorAll("a,button,input,select,textarea").length})'
            )}}

        elif action_type == 'python':
            return {'type': 'observe', 'params': {}, '_fallback_from': 'python'}

        elif action_type == 'delay':
            return {'type': 'delay', 'params': {'duration': 'short'}}

        elif action_type == 'query_agent':
            if grounding_spec is None:
                return {'type': 'observe', 'params': {},
                        '_fallback_from': 'query_agent'}

            # Structural payload: substrate emits its current
            # field state, adapter does wire-format conversion at the
            # boundary. No text rendered here. The peer interprets
            # according to its own faculties.
            objective = [
                {'subject':   a.get('subject', ''),
                 'predicate': a.get('predicate', ''),
                 'object':    a.get('object', ''),
                 'channel_src': a.get('channel_src', ''),
                 'channel_tgt': a.get('channel_tgt', '')}
                for a in self.ledger.semantic_attractors[:5]
                if a.get('basin_type') == 'Objective'
            ]
            subjective = [
                {'subject':   a.get('subject', ''),
                 'predicate': a.get('predicate', ''),
                 'object':    a.get('object', ''),
                 'channel_src': a.get('channel_src', ''),
                 'channel_tgt': a.get('channel_tgt', '')}
                for a in self.ledger.semantic_attractors[:5]
                if a.get('basin_type') == 'Subjective'
            ]
            return {
                'type': 'query_agent',
                'params': {
                    'agent_id':          'default',
                    'desired_delta':     dict(grounding_spec.desired_delta),
                    'top_channels':      list(grounding_spec.top_gradient_channels),
                    'dark_channels':     list(grounding_spec.dark_channels),
                    'attractor_context': {
                        'objective':  objective,
                        'subjective': subjective,
                    },
                    'task':              self._select_library_entry(
                                             self.ledger.task_library, 1) or '',
                    'requester_frame':   self._select_library_entry(
                                             self.ledger.requester_frame, 1) or '',
                },
            }

        elif action_type == 'read_ledger':
            return {'type': 'read_ledger', 'params': {}}

        elif action_type == 'write':
            entry = self._library_entry_from_history('write')
            if entry is None:
                return {'type': 'observe', 'params': {},
                        '_fallback_from': 'write'}
            target = self._select_write_target()
            return {'type': 'write',
                    'params': {'target': target, 'entry': entry}}

        elif action_type == 'write_outreach':
            # Action_dict carries structural payload — no rendered
            # text. The adapter (or downstream peer-typed channel)
            # interprets.
            if grounding_spec is None:
                return {'type': 'write_outreach',
                        'params': {
                            'target':        'operator',
                            'desired_delta': {},
                            'top_channels':  [],
                            'dark_channels': [],
                            'iter':          self._iteration_count,
                        }}
            return {'type': 'write_outreach',
                    'params': {
                        'target':        'operator',
                        'desired_delta': dict(grounding_spec.desired_delta),
                        'top_channels':  list(grounding_spec.top_gradient_channels),
                        'dark_channels': list(grounding_spec.dark_channels),
                        'iter':          self._iteration_count,
                    }}

        elif action_type == 'ground_triplet':
            existing = {(a.get('channel_src', ''), a.get('channel_tgt', ''))
                        for a in self.ledger.semantic_attractors}
            candidates = []
            for (src, tgt), edge in self.state.compression.causal_graph.items():
                if edge.confidence > 0.5 and (src, tgt) not in existing:
                    candidates.append({
                        'subject':      src,
                        'predicate':    'co_moves_with',
                        'object':       tgt,
                        'channel_src':  src,
                        'channel_tgt':  tgt,
                        'label_origin': 'triad',
                    })
                if len(candidates) >= 3:
                    break
            return {'type': 'ground_triplet',
                    'params': {'candidates': candidates}}

        elif action_type == 'revise_triplet':
            if not self.ledger.semantic_attractors:
                return {'type': 'observe', 'params': {},
                        '_fallback_from': 'revise_triplet'}
            lowest_idx = min(
                range(len(self.ledger.semantic_attractors)),
                key=lambda i: self.ledger.semantic_attractors[i].get('confidence', 0.0),
            )
            target = self.ledger.semantic_attractors[lowest_idx]
            edge   = self.state.compression.causal_graph.get(
                (target.get('channel_src', ''), target.get('channel_tgt', ''))
            )
            new_conf = edge.confidence if edge else target.get('confidence', 0.0)
            return {'type': 'revise_triplet',
                    'params': {
                        'triplet_id': lowest_idx,
                        'updates':    {'confidence': new_conf},
                    }}

        elif action_type in ('promote_basin', 'demote_basin'):
            return {'type': action_type, 'params': {}}

        else:
            return {'type': action_type, 'params': {}}

    def _build_affordance_channel(self, affordances: Dict) -> Dict:
        """Build per-affordance channel signals for env_signal.

        Each affordance becomes a sensing channel reporting:
          - coverage:   1.0 if the affordance is currently available, 0.0 if not.
          - magnitude:  1.0 if the affordance was just committed last iteration.
          - rate:       1.0 if just committed (one-iteration spike).

        This is how the substrate perceives "what I can do" and "what I just did."
        Compression's normal co-movement detection then learns affordance→outcome
        edges naturally — actions become first-class causal sources because they
        are sensing channels, not because of any special-case attribution path.
        """
        viable     = set(affordances.get('viable_action_types', set())) if affordances else set()
        just_used  = self._pending_action_attribution
        self._pending_action_attribution = None  # consume; one-iteration semantic

        signals: Dict = {}
        for affordance_name in BASE_AFFORDANCES:
            available  = affordance_name in viable
            was_used   = (affordance_name == just_used)
            signals[affordance_name] = {
                'magnitude': 1.0 if was_used  else 0.0,
                'rate':      1.0 if was_used  else 0.0,
                'coverage':  1.0 if available else 0.0,
            }
        return signals

    def _build_self_channel(self) -> Dict:
        """
        Build self-channel signals carrying Python introspection content.

        Self channels are signal the substrate reads directly without
        transduction — process state, resource use, GC pressure, system
        clock, the iteration cadence of the loop itself. Hierarchical
        'self/...' channel-id prefix marks the class at id level.
        Channels are instantiated dynamically by sensing's signal-arrival
        mechanism — there is no pre-allocated schema; if a new
        introspection signal becomes available it simply emits and the
        channel comes into being.

        magnitude carries the current absolute reading; sensing computes
        last_delta as cross-iteration change.
        """
        signals: Dict = {}

        try:
            wall = time.time()
            signals['self/clock/wall_time'] = {
                'magnitude': float(wall),
                'rate':      1.0,
                'coverage':  1.0,
            }
        except Exception:
            pass

        signals['self/clock/iteration'] = {
            'magnitude': float(self._iteration_count),
            'rate':      1.0,
            'coverage':  1.0,
        }

        # Iteration latency — sensing's last_delta on this channel, after
        # one cross-iteration step, will reflect the time-per-iteration.
        # Compression can discover edges between iteration cost and other
        # channels (e.g., iterations involving agent queries cost more).
        if self._iteration_times:
            last_iter_time = float(self._iteration_times[-1])
            signals['self/clock/iter_latency'] = {
                'magnitude': last_iter_time,
                'rate':      0.0,
                'coverage':  1.0,
            }

        # Process resource utilization. Gracefully degrade if psutil
        # is absent or any specific reading fails on this platform.
        try:
            import psutil
            proc = psutil.Process()
            try:
                cpu = float(proc.cpu_percent(interval=None))
                signals['self/process/cpu_percent'] = {
                    'magnitude': cpu, 'rate': 0.0, 'coverage': 1.0,
                }
            except Exception:
                pass
            try:
                mem = float(proc.memory_info().rss)
                signals['self/process/memory_rss'] = {
                    'magnitude': mem, 'rate': 0.0, 'coverage': 1.0,
                }
            except Exception:
                pass
            try:
                # num_fds is POSIX-only; falls through on Windows.
                fds = float(proc.num_fds())
                signals['self/process/num_fds'] = {
                    'magnitude': fds, 'rate': 0.0, 'coverage': 1.0,
                }
            except Exception:
                pass
        except ImportError:
            pass

        # Python GC counters — three generations of allocation-since-collection
        # counts. last_delta on these tracks GC pressure; a generation-0
        # collection drops gen0_count to 0, which becomes a sharp negative
        # last_delta compression can find structure around.
        try:
            import gc
            counts = gc.get_count()
            for i, c in enumerate(counts):
                signals[f'self/gc/gen{i}_count'] = {
                    'magnitude': float(c), 'rate': 0.0, 'coverage': 1.0,
                }
        except Exception:
            pass

        return signals

    def _build_ledger_channel(self) -> Dict:
        """Substrate-internal "proximity to inherited peak state".

        Computed from coverage overlap between current active channels
        and the operator snapshot persisted in the ledger. Math-spine
        quantities (vol_opt, Φ) are not read; this is a structural
        match between two substrate-level configurations — the
        substrate feeling its similarity to its inherited shape, not
        measuring it against any external observer's quantity.
        """
        op_snap = self.ledger.operator_snapshot
        if not op_snap:
            return {'ledger_proximity': {'magnitude': 0.0, 'rate': 0.0, 'coverage': 0.0}}

        peak_channels = op_snap.get('sensing', {}).get('channels', {})
        if not peak_channels:
            return {'ledger_proximity': {'magnitude': 0.0, 'rate': 0.0, 'coverage': 0.0}}

        current = self.state.sensing.channels
        common = [cid for cid in peak_channels.keys()
                  if cid in current and current[cid].active]
        if not common:
            return {'ledger_proximity': {'magnitude': 0.0, 'rate': 0.0, 'coverage': 0.0}}

        deltas = []
        for cid in common:
            cur_cov  = float(current[cid].coverage)
            peak_cov = float(peak_channels[cid].get('coverage', 0.0))
            deltas.append(abs(cur_cov - peak_cov))
        mean_dist = float(np.mean(deltas))
        proximity = float(np.clip(1.0 - mean_dist, 0.0, 1.0))

        return {'ledger_proximity': {
            'magnitude': proximity,
            'rate':      0.0,
            'coverage':  1.0,
        }}


    def iterate(self) -> None:
        """One iteration of the closure loop S → I → P → A → SMO → S.

        Strict I/O at every operator boundary: each operator's input is
        the prior operator's output. f_rel propagates forward through
        the loop via carriers (compression's active_channel_state,
        prediction's compression attribute, coherence's prediction
        attribute). SMO's emission is the only signal that re-enters
        sensing on the next iteration, and it enters through the same
        env_signal pathway as everything else — no privileged channel.

        Commit gate is coherence.commit_decision: an affordance name
        whose projection extends the running trajectory and which has
        learned outgoing structure in f_rel, or None to hold short.
        No external EOG ranking, no argmax over a quality measure, no
        "stable for N iterations" heuristic — coherence's audit and
        signature_deviation threshold ARE the gate.
        """
        _t0 = time.time()
        self._iteration_count += 1

        self.temporal_memory.decay_all()

        env_signal = self.composite_reality.probe()
        # Fetch affordances once per iteration; reused for sensing's
        # affordance channels and for the post-commit dispatch sanity check.
        current_affordances = self.composite_reality.get_current_affordances()
        env_signal.update(self._build_ledger_channel())
        env_signal.update(self._build_self_channel())
        env_signal.update(self._build_affordance_channel(current_affordances))

        # SMO's relation signals stashed from previous iteration enter
        # sensing through env_signal — the strict-input invariant: sensing
        # has one input shape regardless of whether signals come from
        # self/env probes or from SMO's previous-iteration emission.
        # Sensing computes last_delta = current_magnitude - prior_magnitude
        # on each relation channel, and because SMO's magnitude is the
        # cumulative δf_rel since boot, that last_delta IS this iteration's
        # δf_rel as compression sees it.
        if self._pending_relation_signals:
            env_signal.update(self._pending_relation_signals)
            self._pending_relation_signals = {}

        # ---- S → I → P → A → SMO ----
        # Single positional argument at every boundary.
        new_sensing     = self.state.sensing.apply(env_signal)
        self._sensing_history.append(new_sensing)

        new_compression = self.state.compression.apply(self._sensing_history)
        new_prediction  = self.state.prediction.apply(new_compression)
        new_coherence   = self.state.coherence.apply(new_prediction)

        # Capture prior cumulative_delta before SMO replaces it; the
        # difference (new − prior) IS this iteration's δf_rel — the
        # closure residual the substrate emitted, the substrate's
        # perception of its own motion. Logged on stdout as the loop's
        # structural emission.
        prior_cumulative = dict(self.smo_v151.cumulative_delta)
        new_smo          = self.smo_v151.apply(new_coherence)

        # Stash SMO's emission for next iteration's sensing. SMO emits
        # the relation_signals directly — no _build_relation_channel walk
        # over operator parameters. SMO's output IS the closure-residual
        # signal compression should integrate.
        self._pending_relation_signals = new_smo.relation_signals

        # Update substrate state. SMO's instance carries forward its own
        # cumulative_delta state (per-channel cumulative δf_rel since boot)
        # via reassignment to self.smo_v151 — this is how SMO persists
        # across iterations under the strict-I/O design.
        self.state    = SubstrateState(
            sensing     = new_sensing,
            compression = new_compression,
            prediction  = new_prediction,
            coherence   = new_coherence,
        )
        self.smo_v151 = new_smo

        self._iteration_times.append(time.time() - _t0)

        # δf_rel(this iteration) per channel — non-zero entries only.
        # Computed against substrate emission, not by re-running SMO's
        # internal logic; new_cumulative − prior_cumulative is exactly
        # the bounded_delta SMO clipped this iteration.
        delta_f_rel: Dict[str, float] = {}
        all_channels = set(new_smo.cumulative_delta.keys()) | set(prior_cumulative.keys())
        for cid in all_channels:
            d = (new_smo.cumulative_delta.get(cid, 0.0)
                 - prior_cumulative.get(cid, 0.0))
            if d != 0.0:
                delta_f_rel[cid] = float(d)

        # ---- Commit gate ----
        # coherence.commit_decision is the audit's structural answer:
        # an affordance name whose projection (a) has zero sign-mismatches
        # against running trajectory direction on direction-bearing channels,
        # (b) has learned outgoing structure in f_rel, and was alphabetically
        # first among such candidates. Or None — meaning either the substrate
        # is in flux (signature_deviation above threshold) or no projection
        # extends the trajectory. Either way, no external scoring is needed.
        committed_action: Optional[Dict] = None
        commit = new_coherence.commit_decision
        if commit is not None:
            # Defensive check: between when sensing read affordance availability
            # and now, the reality adapter could have changed which actions are
            # currently executable. If unavailable, skip dispatch; next iteration
            # will re-audit against fresh affordance channels.
            viable_now = (
                set(current_affordances.get('viable_action_types', set()))
                if current_affordances else set()
            )
            if commit in viable_now:
                grounding_spec = self._build_grounding_spec()
                action_dict    = self._action_dict_from_type(
                    commit, current_affordances,
                    check_temporal=True, grounding_spec=grounding_spec,
                )

                self._pending_action_attribution = action_dict.get('type', 'observe')

                pre_ch, post_ch, ctx = self.composite_reality.execute(action_dict)

                if commit == 'query_agent':
                    self._update_agent_registry(action_dict, ctx)

                self._commit_count += 1
                committed_action   = action_dict

        # ---- Structural emission: one JSONL record per iteration ----
        # The substrate's voice. commit is the cause it emitted (the
        # action_dict-as-dispatched, or null). delta_f_rel is the
        # closure residual it perceived as motion this iteration. Math
        # spine readings, summaries, response inspection — all live in
        # the interface layer, which consumes this stream.
        sys.stdout.write(json.dumps({
            'iter':        self._iteration_count,
            't':           time.time(),
            'commit':      committed_action,
            'delta_f_rel': delta_f_rel,
        }, default=_json_default) + '\n')
        sys.stdout.flush()


    def _update_agent_registry(self, action_dict: Dict, ctx: Dict) -> None:
        """Update the per-agent trust signal in the inheritable ledger.

        Trust signal is coherence.loop_closure at the
        moment of commit — the geometric mean of the four pair
        consistencies (s↔i, i↔p, p↔a, smo). High loop_closure at
        commit means the substrate's perception, integration,
        projection, and audit aligned when this agent was queried;
        low means they didn't. This is observational, not optimised
        — the substrate does not select agents to maximize trust;
        coherence's commit_decision selected the affordance, and
        the agent registry just records what coherence's state
        looked like at that moment.
        """
        params      = action_dict.get('params', {})
        agent_id    = params.get('agent_id', 'default')
        context_cfg = params.get('context_configuration', [])

        trust_signal = float(self.state.coherence.consistency.loop_closure)

        registry = dict(self.ledger.agent_registry)
        if agent_id not in registry:
            registry[agent_id] = {
                'agent_id':              agent_id,
                'context_configuration': context_cfg,
                'w_trust':               1.0,
                'last_trust_signal':     trust_signal,
                'trust_history':         [trust_signal],
                'observation_count':     1,
                'is_llm':                True,
            }
        else:
            entry  = dict(registry[agent_id])
            alpha  = 0.1
            w_trust = float((1 - alpha) * entry.get('w_trust', 1.0)
                            + alpha * max(trust_signal, 0.0))
            trust_hist = list(entry.get('trust_history', []))
            trust_hist.append(trust_signal)
            if len(trust_hist) > 20:
                trust_hist = trust_hist[-20:]

            entry['w_trust']               = w_trust
            entry['last_trust_signal']     = trust_signal
            entry['trust_history']         = trust_hist
            entry['observation_count']     = entry.get('observation_count', 0) + 1
            entry['context_configuration'] = context_cfg
            registry[agent_id] = entry

        self.ledger.agent_registry = registry

    def _build_grounding_spec(self) -> Optional[GroundingSpec]:
        """Build the grounding spec the query_agent action uses for
        prompt construction.

        Substrate-internal: uses sensing's channel state, compression's
        causal graph, and per-channel structural mass to identify the
        channels the substrate has the most learned structure about
        (top_gradient_channels) and the under-illuminated channels
        whose signal would deepen compression's discoveries
        (dark_channels). desired_delta is computed from per-channel
        uncertainty. Math-spine quantities are not consulted; the
        substrate shapes its own prompt from its own perception of
        itself.
        """
        active_channels = [
            cid for cid, ch in self.state.sensing.channels.items() if ch.active
        ]
        if not active_channels:
            return None

        # Per-channel structural mass = sum over edges incident on the
        # channel, weighted by |w| · confidence. High mass = the
        # substrate has the most causal structure here.
        channel_mass: Dict[str, float] = {cid: 0.0 for cid in active_channels}
        for (src, tgt), edge in self.state.compression.causal_graph.items():
            w = abs(edge.weight) * edge.confidence
            if src in channel_mass:
                channel_mass[src] += w
            if tgt in channel_mass:
                channel_mass[tgt] += w

        if not channel_mass:
            return None

        top = sorted(active_channels,
                     key=lambda cid: channel_mass.get(cid, 0.0),
                     reverse=True)[:5]

        # Dark = active but low-coverage channels with above-median
        # structural mass. Channels the substrate has learned structure
        # about but is currently receiving little signal from.
        masses      = list(channel_mass.values())
        median_mass = float(np.median(masses)) if masses else 0.0
        dark = [
            cid for cid in active_channels
            if self.state.sensing.channels[cid].coverage < 0.3
            and channel_mass.get(cid, 0.0) > median_mass
        ]

        # desired_delta per channel = (1 - coverage) for active dark/top
        # channels (push toward higher signal level), 0 for others. This is
        # a pure substrate-internal signal: "what would more signal here do."
        # Magnitude scaled by per-channel structural mass so the prompt
        # emphasizes channels the substrate has learned to interpret.
        desired_delta: Dict[str, float] = {}
        max_mass = max(channel_mass.values()) if channel_mass else 1.0
        for cid in active_channels:
            ch = self.state.sensing.channels[cid]
            mass_norm = channel_mass.get(cid, 0.0) / max(max_mass, 1e-9)
            desired_delta[cid] = float((1.0 - ch.coverage) * mass_norm)

        affordances = self.reality.get_current_affordances()

        # Magnitude is the L2 norm of desired_delta — communicates how much
        # the substrate wants to surface per iteration. Substrate-internal,
        # not derived from H⁻¹∇Φ.
        delta_vec = np.array([desired_delta.get(cid, 0.0)
                              for cid in active_channels])
        delta_norm = float(np.linalg.norm(delta_vec))

        return GroundingSpec(
            desired_delta         = desired_delta,
            dark_channels         = dark,
            top_gradient_channels = top,
            current_url           = affordances.get('current_url', ''),
            page_title            = affordances.get('page_title', ''),
            nat_grad_magnitude    = delta_norm,
        )


    def run(self) -> None:
        """Run the closure loop until KeyboardInterrupt.

        No banners, no periodic summaries, no session-end ceremony.
        The substrate's emission is the JSONL stream from iterate();
        ledger persistence happens in the __main__ finally block (or
        in any caller's finally) for graceful Ctrl+C. KeyboardInterrupt
        is swallowed so the interpreter doesn't print a traceback over
        the structural stream.
        """
        try:
            while True:
                self.iterate()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    import sys
    import os

    print('Mentat Triad — UII v19.0', file=sys.stderr)

    if not os.getenv('GROQ_API_KEY'):
        print('FATAL: Set GROQ_API_KEY environment variable.', file=sys.stderr)
        sys.exit(1)

    from groq import Groq

    class GroqAdapter:

        DEFAULT_MODEL = 'llama-3.3-70b-versatile'
        MIN_GAP_S     = 2.1

        def __init__(self, model: Optional[str] = None,
                     max_tokens: int = 512):
            self.client       = Groq(api_key=os.getenv('GROQ_API_KEY'))
            self.model        = model or os.getenv('UII_GROQ_MODEL',
                                                    self.DEFAULT_MODEL)
            self.max_tokens   = max_tokens
            self.last_call    = 0.0
            self.rate_limited = False

        def call(self, prompt: str) -> Tuple[str, int]:
            elapsed = time.time() - self.last_call
            if elapsed < self.MIN_GAP_S:
                time.sleep(self.MIN_GAP_S - elapsed)
            try:
                response = self.client.chat.completions.create(
                    model       = self.model,
                    messages    = [{'role': 'user', 'content': prompt}],
                    temperature = 0.7,
                    max_tokens  = self.max_tokens,
                )
                self.last_call = time.time()
                text   = response.choices[0].message.content or ''
                tokens = response.usage.total_tokens if response.usage else 0
                return text, tokens
            except Exception as e:
                err = str(e) + type(e).__name__
                if '429' in err or 'rate_limit' in err.lower():
                    print('[RATE LIMIT]', file=sys.stderr)
                    self.rate_limited = True
                    return '', 0
                raise

    ledger_path = None
    if '--load-ledger' in sys.argv:
        idx = sys.argv.index('--load-ledger')
        if idx + 1 < len(sys.argv):
            ledger_path = sys.argv[idx + 1]

    llm_client = GroqAdapter()

    ledger = None
    if ledger_path and Path(ledger_path).exists():
        result = load_ledger(ledger_path)
        if result:
            ledger = result[0]
            print(f'[LOADED] Ledger from {ledger_path}', file=sys.stderr)

    reality = BrowserRealityAdapter(base_delta=0.03, headless=True)

    triad = MentatTriad(
        llm_client = llm_client,
        reality    = reality,
        ledger     = ledger,
    )


    DEFAULT_REQUESTER_FRAME = (
        'Responses you give will be used as raw input to a system that '
        'observes patterns over time. Structure responses as observable '
        'facts, not advice or recommendations.'
    )
    if not triad.ledger.requester_frame:
        triad.ledger.requester_frame = [
            os.environ.get('UII_REQUESTER_FRAME', DEFAULT_REQUESTER_FRAME)
        ]

    DEFAULT_TASK = (
        'Describe the structural patterns visible in publicly available '
        'web content the system browses, focusing on what makes pages '
        'navigable and what information can be inferred from layout alone.'
    )
    if not triad.ledger.task_library:
        triad.ledger.task_library = [
            os.environ.get('UII_TASK', DEFAULT_TASK)
        ]

    DEFAULT_OUTPUT_FORMAT = os.environ.get('UII_OUTPUT_FORMAT', '')
    if not triad.ledger.output_format and DEFAULT_OUTPUT_FORMAT:
        triad.ledger.output_format = [DEFAULT_OUTPUT_FORMAT]

    if 'operator' not in triad.ledger.agent_registry:
        triad.ledger.agent_registry['operator'] = {
            'is_llm':        False,
            'mechanism':     'unimplemented',
            'observation_count': 0,
            'w_trust':       1.0,
            'context_configuration': [],
        }

    try:
        triad.run()
    finally:
        save_ledger(triad.ledger, ledger_path or 'ledger.json')
        print(f'Ledger saved → {ledger_path or "ledger.json"}', file=sys.stderr)
        reality.close()

