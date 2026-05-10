"""uii_reality.py — UII v19.0 — perturbation surface: reality adapters.

Reality adapters are the substrate's interface to perturbation sources
outside the closure loop. Each adapter implements probe() (returns a
flat dict of {channel_id: {magnitude, rate, coverage}} measurements),
execute(action_dict) (commits an action and returns
(pre_channels, post_channels, context)), and get_current_affordances()
(reports what's currently executable).

Three adapters are provided:
  - BrowserRealityAdapter:    Playwright-driven DOM perturbation; env
                              channels for browser/dom_complexity/etc.
  - AgentRealityAdapter:      LLM-mediated perturbation; env channels
                              for agent_response_latency, agent_token_count,
                              agent_availability.
  - CompositeRealityAdapter:  Wraps Browser + Agent into one adapter
                              with action-type routing; also handles
                              relation-class actions (ground_triplet,
                              revise_triplet, promote_basin,
                              demote_basin) and self-class actions
                              (read_ledger, write, write_outreach).

The substrate sees no architectural distinction between channel
classes (self / env / affordance / relation) — they all enter sensing
through env_signal as a flat dict and compression's edge formation
treats them identically. The class structure here is convention only,
visible in channel_id naming and where the signals come from.

Relation to the math spine: reality adapters produce the per-channel
magnitude readings sensing computes last_delta from. None of the math
spine quantities (Φ, ∇Φ, Hessian, vol_opt) are computed inside reality
adapters; they're computed externally from the substrate's emitted
operator state. AttractorMonitor's basin classification reads
ledger.hessian_snapshot but does not modify the loop.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Callable
import numpy as np
import copy
import os
import time
import json
from collections import deque

from uii_geometry import (
    BASE_AFFORDANCES, SUBSTRATE_DIMS,
    SubstrateState, StateTrace,
    RealityAdapter,
    AgentHandler, AVAILABLE_AGENTS,
)


def _build_channel_probes() -> Dict[str, Callable]:
    import gc as _gc
    import os as _os
    import sys as _sys

    probes: Dict[str, Callable] = {}

    probes['clock']          = lambda: {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
    probes['clock_rate']     = lambda: {'magnitude': 1.0, 'rate': 1.0, 'coverage': 1.0}
    probes['os_signals']     = lambda: {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
    probes['entropy_source'] = lambda: {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
    probes['env_vars']       = lambda: {
        'magnitude': min(len(_os.environ) / 100.0, 1.0), 'rate': 0.0, 'coverage': 1.0,
    }
    probes['gc_pressure']    = lambda: (lambda c: {
        'magnitude': min((c[0] / 700.0 + c[1] / 70.0 + c[2] / 10.0) / 3.0, 1.0),
        'rate': 1.0, 'coverage': 1.0,
    })(_gc.get_count())

    def _stdin() -> Optional[Dict]:
        try:
            import select
            ready = bool(select.select([_sys.stdin], [], [], 0.0)[0])
            return {'magnitude': float(ready), 'rate': float(ready), 'coverage': 1.0}
        except Exception:
            return None
    probes['stdin'] = _stdin

    try:
        import psutil as _ps
        _proc = _ps.Process()

        probes['resource_cpu']     = lambda: {'magnitude': _ps.cpu_percent() / 100.0,            'rate': 1.0, 'coverage': 1.0}
        probes['resource_memory']  = lambda: {'magnitude': _ps.virtual_memory().percent / 100.0,  'rate': 1.0, 'coverage': 1.0}
        probes['resource_swap']    = lambda: {'magnitude': _ps.swap_memory().percent / 100.0,     'rate': 1.0, 'coverage': 1.0}
        probes['resource_disk']    = lambda: {'magnitude': _ps.disk_usage('/').percent / 100.0,   'rate': 1.0, 'coverage': 1.0}
        probes['process_self']     = lambda: {'magnitude': _proc.memory_percent() / 100.0,        'rate': 1.0, 'coverage': 1.0}
        probes['process_threads']  = lambda: {'magnitude': min(_proc.num_threads() / 50.0, 1.0),  'rate': 1.0, 'coverage': 1.0}
        probes['process_children'] = lambda: {
            'magnitude': min(len(_proc.children()) / 10.0, 1.0),
            'rate':      float(len(_proc.children()) > 0),
            'coverage':  1.0,
        }

        def _fd() -> Optional[Dict]:
            try:
                return {'magnitude': min(_proc.num_fds() / 1024.0, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except AttributeError:
                return None
        probes['resource_fd'] = _fd

        def _thermal() -> Optional[Dict]:
            try:
                t = _ps.sensors_temperatures()
                if t:
                    vals = [x.current for readings in t.values() for x in readings]
                    return {'magnitude': min(max(vals) / 100.0, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['resource_thermal'] = _thermal

        def _battery() -> Optional[Dict]:
            try:
                b = _ps.sensors_battery()
                if b:
                    return {
                        'magnitude': b.percent / 100.0,
                        'rate':      float(not b.power_plugged),
                        'coverage':  1.0,
                    }
            except Exception:
                pass
            return None
        probes['resource_battery'] = _battery

        def _net_if() -> Optional[Dict]:
            try:
                stats = _ps.net_if_stats()
                up = sum(1 for s in stats.values() if s.isup)
                return {'magnitude': up / max(len(stats), 1), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                return None
        probes['network_interface'] = _net_if

        def _net_bw() -> Optional[Dict]:
            try:
                c = _ps.net_io_counters()
                if c:
                    return {'magnitude': min((c.bytes_sent + c.bytes_recv) / 1e9, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['network_bandwidth'] = _net_bw

        def _disk_io() -> Optional[Dict]:
            try:
                c = _ps.disk_io_counters()
                if c:
                    return {'magnitude': min((c.read_bytes + c.write_bytes) / 1e9, 1.0), 'rate': 1.0, 'coverage': 1.0}
            except Exception:
                pass
            return None
        probes['filesystem_io'] = _disk_io

        def _syslog() -> Optional[Dict]:
            import os as _o
            path = '/var/log/syslog'
            if _o.path.exists(path):
                return {'magnitude': min(_o.path.getsize(path) / 1e7, 1.0), 'rate': 1.0, 'coverage': 1.0}
            return None
        probes['system_logs'] = _syslog

    except ImportError:
        pass

    return probes


_CHANNEL_PROBES: Dict[str, Callable] = _build_channel_probes()


class AttractorMonitor:
    """Classifies semantic_attractors basin type (Objective/Subjective) from causal graph evidence."""

    CONFIDENCE_THRESHOLD = 0.5
    VARIANCE_THRESHOLD   = 0.3
    VOL_OPT_THRESHOLD    = 1.0

    def assess(self,
               channel_src: str,
               channel_tgt: str,
               compression,
               ledger) -> str:
        edge = compression.causal_graph.get((channel_src, channel_tgt))
        if edge is None or edge.confidence < self.CONFIDENCE_THRESHOLD:
            return 'Subjective'

        hs = ledger.hessian_snapshot
        if not hs or hs.get('vol_opt', 0.0) < self.VOL_OPT_THRESHOLD:
            return 'Subjective'

        sig_dev = ledger.operator_snapshot.get('coherence', {}).get(
            'signature_deviation', 1.0
        )
        if sig_dev > self.VARIANCE_THRESHOLD:
            return 'Subjective'

        return 'Objective'

    def reclassify_all(self, semantic_attractors: List,
                        compression, ledger) -> List:
        result = []
        for attractor in semantic_attractors:
            basin_type = self.assess(
                attractor.get('channel_src', ''),
                attractor.get('channel_tgt', ''),
                compression,
                ledger,
            )
            result.append({**attractor, 'basin_type': basin_type})
        return result


LATENCY_REFERENCE_S = 10.0
TOKEN_REFERENCE     = 1000


class AgentRealityAdapter:
    """LLM as substrate: latency/tokens/availability via event-buffer aggregation between probes."""

    def __init__(self, llm_client):
        self.llm_client       = llm_client
        self._last_channels:  Dict = {}
        self._response_history: deque = deque(maxlen=50)

        self._events_since_last_probe: deque = deque(maxlen=100)

        self._prev_latency:   float = 0.0
        self._prev_tokens:    float = 0.0
        self._prev_available: float = 0.0

        self._ever_called:    bool  = False
        self._cumulative_tokens: int = 0


    def probe(self) -> Dict:
        if not self._ever_called:
            return {}

        events = list(self._events_since_last_probe)
        self._events_since_last_probe.clear()

        if events:
            cur_latency   = sum(e['latency']   for e in events) / len(events)
            cur_tokens    = sum(e['tokens']    for e in events) / len(events)
            cur_available = sum(e['available'] for e in events) / len(events)
        else:
            cur_latency   = self._prev_latency
            cur_tokens    = self._prev_tokens
            cur_available = self._prev_available

        lat_mag      = float(np.clip(cur_latency / LATENCY_REFERENCE_S, 0.0, 1.0))
        tok_mag      = float(np.clip(cur_tokens  / TOKEN_REFERENCE,     0.0, 1.0))
        prev_lat_mag = float(np.clip(self._prev_latency / LATENCY_REFERENCE_S, 0.0, 1.0))
        prev_tok_mag = float(np.clip(self._prev_tokens  / TOKEN_REFERENCE,     0.0, 1.0))

        result = {
            'agent_response_latency': {
                'magnitude': lat_mag,
                'rate':      lat_mag - prev_lat_mag,
                'coverage':  1.0,
            },
            'agent_token_count': {
                'magnitude': tok_mag,
                'rate':      tok_mag - prev_tok_mag,
                'coverage':  1.0,
            },
            'agent_availability': {
                'magnitude': cur_available,
                'rate':      cur_available - self._prev_available,
                'coverage':  1.0,
            },
        }

        if events:
            self._prev_latency   = cur_latency
            self._prev_tokens    = cur_tokens
            self._prev_available = cur_available

        return result


    def execute(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict]:
        params      = action_dict.get('params', {})
        agent_id    = params.get('agent_id', 'default')

        # Substrate emits structural payload; adapter does
        # wire-format conversion at the boundary. The structural payload
        # is JSON-encoded as-is; if the LLM needs orientation on the
        # format, that's a one-shot system prompt set at session start
        # by the llm_client, not a wrapper around each call.
        wire_payload = json.dumps({
            'agent_id':          agent_id,
            'desired_delta':     params.get('desired_delta', {}),
            'top_channels':      params.get('top_channels', []),
            'dark_channels':     params.get('dark_channels', []),
            'attractor_context': params.get('attractor_context',
                                             {'objective': [], 'subjective': []}),
            'task':              params.get('task', ''),
            'requester_frame':   params.get('requester_frame', ''),
        }, default=str)

        pre_channels = self.probe()

        t_start = time.time()
        try:
            response, tokens_used = self.llm_client.call(wire_payload)
            latency   = time.time() - t_start
            available = 1.0
        except Exception:
            response    = ''
            tokens_used = 0
            latency     = time.time() - t_start
            available   = 0.0

        self._events_since_last_probe.append({
            'latency':   latency,
            'tokens':    tokens_used,
            'available': available,
        })
        self._ever_called = True

        self._cumulative_tokens += int(tokens_used)

        if response:
            self._response_history.append(response)

        post_channels = self.probe()

        context = {
            'agent_id':           agent_id,
            'wire_payload':       wire_payload,
            'raw_response':       response,
            'response_latency_s': latency,
            'tokens_used':        tokens_used,
            'agent_available':    available > 0.5,
            'action_succeeded':   available > 0.5,
            'refusal':            False,
        }

        self._last_channels = post_channels
        return pre_channels, post_channels, context


    def get_current_affordances(self) -> Dict:
        return {
            'agent_available': self._ever_called and self._prev_available > 0.5,
        }


# UserRealityAdapter, peer to BrowserRealityAdapter and
# AgentRealityAdapter. The user becomes another adaptive system the
# substrate metabolizes through ordinary perturbation, on equal terms
# with browser and LLM. Channel signals describe input economics
# (availability, content size, inter-input latency); the substrate's
# compression discovers the user's economic profile through ordinary
# cross-iteration edge formation, just like it does for browser and
# agent peers. No substrate-level peer typing; the differentiation is
# emergent.

USER_CONTENT_SIZE_REFERENCE = 500    # chars per input
USER_AVAILABILITY_DECAY_S   = 30.0   # seconds for availability to decay 1 → 0


class UserRealityAdapter:
    """User as peer adaptive system.

    Forward direction: the interface (forward transducer) calls
    push_input(text) to enqueue a user perturbation. probe() drains
    pending inputs at sensing time and emits user/* channels with the
    same shape as agent/* and browser channels — sensing metabolizes
    them on equal footing. Substrate has no awareness of how the queue
    is populated; it sees channels.

    Reverse direction: execute() receives user-targeted action_dicts
    (when the substrate commits to a user-directed peer action) and
    pushes the structural payload to the outbound queue for the
    interface to render. The interface drains via drain_outbound().

    Channel design:
      user/availability   — 1.0 when input arrived recently, decays to 0
      user/content_size   — normalized chars-per-input
      user/input_latency  — time between consecutive inputs, normalized

    No substrate-level peer typing. The user is just another peer with
    its own economic profile (different latency distribution, different
    cost zero to api_llm budget, different availability pattern).
    Compression's learned f_rel discovers what the user actually is
    through cross-class edge formation."""

    def __init__(self):
        # Forward queue: interface pushes here, probe() drains.
        self._pending_inputs:        deque = deque()
        # Reverse queue: execute() pushes here, interface drains.
        self._outbound_actions:      deque = deque()

        # Rolling buffer of received events (for windowed reads).
        self._events_since_last_probe: deque = deque(maxlen=100)

        self._prev_content_size:  float = 0.0
        self._prev_input_latency: float = 0.0
        self._prev_availability:  float = 0.0

        self._last_input_time: Optional[float] = None
        self._ever_received:   bool            = False

    # ---- Forward transducer entry point ----
    def push_input(self, text: str, timestamp: Optional[float] = None) -> None:
        """Called by the interface (forward transducer) to enqueue a
        user perturbation. Pure transduction: depends only on user-side
        state. Substrate metabolizes on next probe() — no synchronous
        coupling, no return-of-handshake."""
        ts = timestamp if timestamp is not None else time.time()
        self._pending_inputs.append({'text': text, 'timestamp': ts})

    # ---- Reverse transducer drain ----
    def drain_outbound(self) -> List[Dict]:
        """Called by the interface (reverse transducer for outbound
        actions) to retrieve any user-targeted action payloads the
        substrate has emitted. Returns and clears the queue. Note:
        observability is JSONL-only and entirely separate from this
        path; this drain is for the substrate's *outbound peer
        action* payloads to the user, not for trajectory observation."""
        out = list(self._outbound_actions)
        self._outbound_actions.clear()
        return out

    # ---- Sensing-side metabolization ----
    def probe(self) -> Dict:
        if not self._ever_received and not self._pending_inputs:
            return {}

        # Drain pending inputs into events for this iteration.
        events: List[Dict] = []
        while self._pending_inputs:
            ev = self._pending_inputs.popleft()
            events.append(ev)
            self._events_since_last_probe.append(ev)
            self._ever_received = True

        now = time.time()

        if events:
            cur_content_size = sum(len(e['text']) for e in events) / len(events)

            if self._last_input_time is not None:
                latencies = []
                prev_t = self._last_input_time
                for e in events:
                    latencies.append(max(0.0, e['timestamp'] - prev_t))
                    prev_t = e['timestamp']
                cur_input_latency = sum(latencies) / len(latencies)
            else:
                cur_input_latency = 0.0

            self._last_input_time = events[-1]['timestamp']
            cur_availability      = 1.0
        else:
            cur_content_size  = self._prev_content_size
            cur_input_latency = self._prev_input_latency
            if self._last_input_time is not None:
                elapsed = now - self._last_input_time
                cur_availability = float(np.clip(
                    1.0 - elapsed / USER_AVAILABILITY_DECAY_S, 0.0, 1.0
                ))
            else:
                cur_availability = 0.0

        size_mag      = float(np.clip(cur_content_size  / USER_CONTENT_SIZE_REFERENCE, 0.0, 1.0))
        lat_mag       = float(np.clip(cur_input_latency / LATENCY_REFERENCE_S,         0.0, 1.0))
        avail_mag     = float(np.clip(cur_availability,                                0.0, 1.0))
        prev_size_mag = float(np.clip(self._prev_content_size  / USER_CONTENT_SIZE_REFERENCE, 0.0, 1.0))
        prev_lat_mag  = float(np.clip(self._prev_input_latency / LATENCY_REFERENCE_S,         0.0, 1.0))

        result = {
            'user/availability': {
                'magnitude': avail_mag,
                'rate':      avail_mag - self._prev_availability,
                'coverage':  1.0,
            },
            'user/content_size': {
                'magnitude': size_mag,
                'rate':      size_mag - prev_size_mag,
                'coverage':  1.0 if events else 0.5,
            },
            'user/input_latency': {
                'magnitude': lat_mag,
                'rate':      lat_mag - prev_lat_mag,
                'coverage':  1.0,
            },
        }

        if events:
            self._prev_content_size  = cur_content_size
            self._prev_input_latency = cur_input_latency
        self._prev_availability = cur_availability

        return result

    # ---- Outbound action handler ----
    def execute(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict]:
        """Receives user-targeted action_dicts and pushes the structural
        payload to the outbound queue. Action types don't currently
        route here; included for architectural completeness so the
        peer-adapter pattern is structurally symmetric."""
        params = action_dict.get('params', {})

        pre_channels = self.probe()

        self._outbound_actions.append({
            'action_dict': action_dict,
            'timestamp':   time.time(),
        })

        post_channels = self.probe()

        context = {
            'action_type':      action_dict.get('type'),
            'agent_id':         params.get('agent_id', 'user'),
            'action_succeeded': True,
            'refusal':          False,
        }
        return pre_channels, post_channels, context

    def get_current_affordances(self) -> Dict:
        return {
            'user_available':      self._ever_received and self._prev_availability > 0.0,
            'user_pending_inputs': len(self._pending_inputs),
        }


from uii_geometry import RealityAdapter as _RealityAdapter


class CompositeRealityAdapter(_RealityAdapter):
    """Wraps Browser+Agent+User; flat probe() dict, execute() routes by action type."""

    def __init__(self,
                 browser:           'BrowserRealityAdapter',
                 agent:             AgentRealityAdapter,
                 ledger,
                 get_state:         Callable,
                 attractor_monitor: AttractorMonitor,
                 user:              Optional[UserRealityAdapter] = None):
        self.browser           = browser
        self.agent             = agent
        self.user              = user
        self.ledger            = ledger
        self.get_state         = get_state
        self.attractor_monitor = attractor_monitor


    def probe(self) -> Dict:
        channels = {}
        channels.update(self.browser.probe())
        channels.update(self.agent.probe())
        if self.user is not None:
            channels.update(self.user.probe())

        try:
            budget = int(os.environ.get('UII_API_LLM_BUDGET', '100000'))
        except ValueError:
            budget = 100000

        cumulative = self.agent._cumulative_tokens

        agent_tok = channels.get('agent_token_count', {'magnitude': 0.0, 'rate': 0.0})
        magnitude = float(agent_tok['magnitude'])
        rate      = float(agent_tok['rate'])
        coverage  = float(np.clip(1.0 - cumulative / max(budget, 1), 0.0, 1.0))

        channels['api_llm'] = {
            'magnitude': magnitude,
            'rate':      rate,
            'coverage':  coverage,
        }

        return channels


    def execute(self, action_dict: Dict,
                boundary_pressure:   float = 0.0) -> Tuple[Dict, Dict, Dict]:
        action_type = action_dict.get('type', 'observe')

        if action_type == 'query_agent':
            return self.agent.execute(action_dict)

        if action_type in ('ground_triplet', 'revise_triplet',
                           'promote_basin', 'demote_basin'):
            return self._execute_relation(action_dict)

        if action_type in ('read_ledger', 'write', 'write_outreach'):
            return self._execute_self(action_dict)

        return self.browser.execute(action_dict, boundary_pressure)


    def _execute_relation(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict]:
        action_type = action_dict.get('type')
        params      = action_dict.get('params', {})
        state       = self.get_state()

        pre_attractor_count = len(self.ledger.semantic_attractors)
        pre_channels = {
            'ledger_proximity': {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
        }

        if action_type == 'ground_triplet':
            candidates = params.get('candidates', [])
            new_attractors = []
            for candidate in candidates:
                src = candidate.get('channel_src', '')
                tgt = candidate.get('channel_tgt', '')
                edge = state.compression.causal_graph.get((src, tgt))
                confidence = edge.confidence if edge else 0.1

                basin_type = self.attractor_monitor.assess(
                    src, tgt, state.compression, self.ledger
                )

                triplet = {
                    'subject':      candidate.get('subject', ''),
                    'predicate':    candidate.get('predicate', ''),
                    'object':       candidate.get('object', ''),
                    'confidence':   confidence,
                    'basin_type':   basin_type,
                    'channel_src':  src,
                    'channel_tgt':  tgt,
                    'label_origin': candidate.get('label_origin', 'llm'),
                }
                new_attractors.append(triplet)

            self.ledger.semantic_attractors.extend(new_attractors)

        elif action_type == 'revise_triplet':
            triplet_id = params.get('triplet_id')
            updates    = params.get('updates', {})
            if (triplet_id is not None and
                    triplet_id < len(self.ledger.semantic_attractors)):
                existing = self.ledger.semantic_attractors[triplet_id]
                self.ledger.semantic_attractors[triplet_id] = {
                    **existing, **updates
                }

        elif action_type in ('promote_basin', 'demote_basin'):
            self.ledger.semantic_attractors = self.attractor_monitor.reclassify_all(
                self.ledger.semantic_attractors, state.compression, self.ledger
            )

        post_attractor_count = len(self.ledger.semantic_attractors)
        delta_count = post_attractor_count - pre_attractor_count

        post_channels = {
            'ledger_proximity': {
                'magnitude': float(np.clip(delta_count / 10.0, 0.0, 1.0)),
                'rate':      float(np.clip(delta_count / 10.0, -1.0, 1.0)),
                'coverage':  1.0,
            }
        }

        context = {
            'action_type':       action_type,
            'attractors_before': pre_attractor_count,
            'attractors_after':  post_attractor_count,
            'action_succeeded':  True,
            'refusal':           False,
        }
        return pre_channels, post_channels, context


    def _execute_self(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict]:
        action_type = action_dict.get('type')
        params      = action_dict.get('params', {})

        pre_channels = {
            'ledger_proximity': {'magnitude': 0.0, 'rate': 0.0, 'coverage': 1.0}
        }

        succeeded = False

        if action_type == 'read_ledger':
            succeeded = True

        elif action_type == 'write':
            target = params.get('target', '')
            entry  = params.get('entry', '')
            if entry:
                if target == 'task_library':
                    self.ledger.task_library.append(entry)
                    succeeded = True
                elif target == 'requester_frame':
                    self.ledger.requester_frame.append(entry)
                    succeeded = True
                elif target == 'output_format':
                    self.ledger.output_format.append(entry)
                    succeeded = True

        elif action_type == 'write_outreach':
            target = params.get('target', '')
            # write_outreach carries structural payload, not text.
            # Log the structural shape for later inspection; the substrate
            # has emitted its current field state, the outreach record
            # captures what was emitted.
            structural = {
                'desired_delta': params.get('desired_delta', {}),
                'top_channels':  params.get('top_channels', []),
                'dark_channels': params.get('dark_channels', []),
                'iter':          params.get('iter', None),
            }
            if structural['top_channels'] or structural['desired_delta']:
                causal = dict(self.ledger.causal_model)
                outreach = causal.get('outreach_log', [])
                outreach.append({
                    'structural': structural,
                    'target':     target,
                    'timestamp':  time.time(),
                    'mechanism':  'unimplemented',
                })
                causal['outreach_log'] = outreach
                self.ledger.causal_model = causal
                succeeded = True

        post_channels = {
            'ledger_proximity': {
                'magnitude': 1.0 if succeeded else 0.0,
                'rate':      1.0 if succeeded else 0.0,
                'coverage':  1.0,
            }
        }

        context = {
            'action_type':      action_type,
            'action_succeeded': succeeded,
            'refusal':          False,
        }
        return pre_channels, post_channels, context


    def get_current_affordances(self) -> Dict:
        affordances = self.browser.get_current_affordances()
        affordances['agent_available'] = self.agent.get_current_affordances().get(
            'agent_available', False
        )
        if self.user is not None:
            affordances.update(self.user.get_current_affordances())
        return affordances

    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]:
        return self.browser.execute_trajectory(trajectory)

    def close(self):
        self.browser.close()


class BrowserRealityAdapter(RealityAdapter):
    """Playwright DOM perturbations; affordances surfaced per probe."""

    def __init__(self, base_delta: float = 0.03, headless: bool = True,
                 start_url: str = 'https://github.com'):
        self.base_delta = base_delta
        self.headless = headless
        self.start_url = start_url

        self.initialized: bool = False
        self._ever_navigated: bool = False

        self.volatility_history: deque = deque(maxlen=10)

        from playwright.sync_api import sync_playwright
        self._init_browser()

    def _init_browser(self):
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(viewport={'width': 1280, 'height': 720})
        self.page = self.context.new_page()

        try:
            self.page.goto(self.start_url, wait_until='networkidle', timeout=15000)
            self._ever_navigated = True
        except Exception as e:
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                self._ever_navigated = True
            except Exception:
                print(f"[REALITY] Warning: start_url navigation failed ({e}). "
                      f"Browser ready — CNS must navigate before links are available.")

        self.initialized = True

    def probe(self) -> Dict:
        return self.measure_channels(context=None)

    def get_current_affordances(self) -> Dict:
        try:
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=1000)
            except Exception:
                pass

            current_url = self.page.url

            if current_url == 'about:blank' and not self._ever_navigated:
                return {
                    'links': [],
                    'buttons': [],
                    'inputs': [],
                    'readable': [],
                    'current_url': 'about:blank',
                    'page_title': '',
                    'scroll_position': 0,
                    'viewport_height': 0,
                    'total_height': 0
                }

            if current_url != 'about:blank':
                self._ever_navigated = True

            affordances = self.page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({
                        url: a.href,
                        text: a.innerText.trim().slice(0, 100),
                        visible: (a.offsetWidth > 0 || a.offsetHeight > 0)
                    }))
                    .filter(l =>
                        l.visible &&
                        l.url.startsWith('http') &&
                        l.url !== window.location.href
                    )
                    .slice(0, 50);

                const buttons = Array.from(document.querySelectorAll(
                    'button, [role="button"], input[type="submit"], input[type="button"]'
                ))
                    .map((b, i) => ({
                        selector: b.id ? `#${b.id}` : `${b.tagName.toLowerCase()}:nth-of-type(${i+1})`,
                        text: b.innerText || b.value || '',
                        visible: b.offsetParent !== null
                    }))
                    .filter(b => b.visible)
                    .slice(0, 30);

                const inputs = Array.from(document.querySelectorAll(
                    'input:not([type="submit"]):not([type="button"]), textarea, select'
                ))
                    .map((inp, i) => ({
                        selector: inp.id ? `#${inp.id}` : inp.name ? `[name="${inp.name}"]` : `input:nth-of-type(${i+1})`,
                        type: inp.type || inp.tagName.toLowerCase(),
                        placeholder: inp.placeholder || '',
                        visible: inp.offsetParent !== null
                    }))
                    .filter(inp => inp.visible)
                    .slice(0, 20);

                const readable = Array.from(document.querySelectorAll(
                    'article, main, [role="main"], .content, #content'
                ))
                    .map((el, i) => ({
                        selector: el.id ? `#${el.id}` : el.className ? `.${el.className.split(' ')[0]}` : `article:nth-of-type(${i+1})`,
                        preview: el.innerText.slice(0, 200)
                    }))
                    .slice(0, 10);

                return {
                    links: links,
                    buttons: buttons,
                    inputs: inputs,
                    readable: readable,
                    current_url: window.location.href,
                    page_title: document.title,
                    scroll_position: window.scrollY,
                    viewport_height: window.innerHeight,
                    total_height: document.documentElement.scrollHeight
                };
            }""")

            return affordances

        except Exception as e:
            return {
                'links': [],
                'buttons': [],
                'inputs': [],
                'readable': [],
                'current_url': '',
                'page_title': '',
                'scroll_position': 0,
                'viewport_height': 0,
                'total_height': 0
            }

    def _measure_dom_state(self) -> Dict:
        try:
            metrics = self.page.evaluate("""() => {
                return {
                    text_length: document.body.innerText.length,
                    link_count: document.querySelectorAll('a').length,
                    image_count: document.querySelectorAll('img').length,
                    input_count: document.querySelectorAll('input, textarea, select').length,
                    dom_depth: (function() {
                        let maxDepth = 0;
                        function getDepth(element, depth) {
                            maxDepth = Math.max(maxDepth, depth);
                            for (let child of element.children) {
                                getDepth(child, depth + 1);
                            }
                        }
                        getDepth(document.body, 0);
                        return maxDepth;
                    })(),
                    element_count: document.querySelectorAll('*').length,
                    interactive_count: document.querySelectorAll('a, button, input, select, textarea').length,
                    form_count: document.querySelectorAll('form').length,
                    has_errors: document.querySelectorAll('[class*="error"], [id*="error"]').length > 0,
                    scroll_height: document.documentElement.scrollHeight,
                    viewport_height: window.innerHeight,
                    url: window.location.href,
                    title: document.title
                };
            }""")
            return metrics
        except Exception as e:
            return {
                'text_length': 0, 'link_count': 0, 'image_count': 0,
                'input_count': 0, 'dom_depth': 0, 'element_count': 0,
                'interactive_count': 0, 'form_count': 0, 'has_errors': False,
                'scroll_height': 0, 'viewport_height': 0, 'url': '', 'title': ''
            }

    def measure_channels(self, context: Dict = None) -> Dict:
        import os
        signal: Dict = {}

        for cid, probe in _CHANNEL_PROBES.items():
            try:
                result = probe()
                if result is not None:
                    signal[cid] = result
            except Exception:
                pass

        try:
            dom = self._measure_dom_state()
            elem_count           = max(dom.get('element_count', 1), 1)
            interactive_fraction = dom.get('interactive_count', 0) / elem_count
            viewport_fraction    = min(
                1.0,
                dom.get('viewport_height', 0) / max(dom.get('scroll_height', 1), 1)
            )
            signal['browser'] = {
                'magnitude': float(np.clip(interactive_fraction, 0.0, 1.0)),
                'rate':      float(np.clip(interactive_fraction, 0.0, 1.0)),
                'coverage':  float(np.clip(viewport_fraction,    0.0, 1.0)),
            }
        except Exception:
            pass

        return signal

    def _classify_error(self, e: Exception) -> str:
        msg = str(e).lower()
        if '429' in msg or 'rate limit' in msg:
            return 'rate_limit'
        if 'token' in msg and ('limit' in msg or 'quota' in msg):
            return 'token_exhaustion'
        if 'timeout' in msg:
            return 'timeout'
        return 'unknown'

    def execute(self, action: Dict, boundary_pressure: float = 0.0) -> Tuple[Dict, Dict, Dict]:
        t_start = time.time()

        action_type = action.get('type', 'observe')
        params = action.get('params', {})

        before_metrics = self._measure_dom_state()
        pre_channels   = self.measure_channels()
        action_succeeded = True

        try:
            if action_type == 'navigate':
                url = params.get('url')
                if not url:
                    raise ValueError("Navigate requires 'url'")
                try:
                    self.page.goto(url, wait_until='networkidle', timeout=10000)
                except Exception:
                    try:
                        self.page.wait_for_load_state('domcontentloaded', timeout=3000)
                    except Exception:
                        pass

            elif action_type == 'click':
                selector = params.get('selector', 'a')
                self.page.click(selector, timeout=3000)

            elif action_type == 'fill':
                selector = params.get('selector', 'input')
                text = params.get('text', '')
                self.page.fill(selector, text, timeout=3000)

            elif action_type == 'type':
                selector = params.get('selector', 'input')
                text = params.get('text', '')
                self.page.type(selector, text, timeout=3000)

            elif action_type == 'evaluate':
                script = params.get('script', '')
                result = self.page.evaluate(script)

            elif action_type == 'read':
                selector = params.get('selector', 'body')
                content = self.page.locator(selector).text_content(timeout=3000)

            elif action_type == 'scroll':
                direction = params.get('direction', 'down')
                amount = params.get('amount', 300)
                if direction == 'down':
                    self.page.evaluate(f"window.scrollBy(0, {amount})")
                else:
                    self.page.evaluate(f"window.scrollBy(0, -{amount})")

            elif action_type == 'observe':
                self.page.wait_for_timeout(100)

            elif action_type == 'delay':
                duration = params.get('duration', 'short')
                wait_time = {'short': 500, 'medium': 1500, 'long': 3000}.get(duration, 500)
                self.page.wait_for_timeout(wait_time)

            elif action_type == 'python':
                return self._execute_python(params, before_metrics)

            elif action_type == 'write_file':
                path    = params.get('path', '')
                content = params.get('content', '')
                if path and content:
                    try:
                        with open(path, 'w') as _f:
                            _f.write(content)
                    except Exception:
                        pass

            elif action_type == 'send_email':
                pass

            else:
                pass

            self.page.wait_for_timeout(200)

        except Exception as e:
            error_type = self._classify_error(e)

            if error_type in ['rate_limit', 'token_exhaustion', 'timeout']:
                error_context = {
                    'refusal': True,
                    'recoverable': error_type != 'token_exhaustion',
                    'reason': error_type,
                    'interaction_surface_available': error_type == 'timeout',
                    'boundary_pressure': boundary_pressure,
                    'before': before_metrics,
                    'after': before_metrics,
                    'response_latency_ms': (time.time() - t_start) * 1000,
                }
                return pre_channels, pre_channels, error_context

            action_succeeded = False

        response_latency_ms = (time.time() - t_start) * 1000
        after_metrics = self._measure_dom_state()

        context = {
            'before': before_metrics,
            'after': after_metrics,
            'action_succeeded': action_succeeded,
            'refusal': False,
            'boundary_pressure': boundary_pressure,
            'url_changed': before_metrics['url'] != after_metrics['url'],
            'new_url': after_metrics['url'],
            'page_title': after_metrics['title'],
            'response_latency_ms': response_latency_ms,
        }

        post_channels = self.measure_channels(context=context)
        return pre_channels, post_channels, context
    
    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]:
        perturbation_trace = []
        try:
            for step in trajectory:
                delta, context = self.execute(step)
                perturbation_trace.append({
                    'action': step,
                    'delta':  delta,
                    'context': context,
                })
            return perturbation_trace, True
        except Exception as e:
            return perturbation_trace, False
    

    def _execute_python(self, params: Dict, before_metrics: Dict) -> Tuple[Dict, Dict]:
        import os

        code = params.get('code')
        if not code:
            raise ValueError("python affordance requires 'code' parameter")

        cwd = os.getcwd()

        exec_globals = {
            '__builtins__': __builtins__,
            'cwd': cwd,
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            result = exec_locals.get('result', None)

            post_channels = self.measure_channels()
            return (
                self.measure_channels(),
                post_channels,
                {
                    'before': before_metrics,
                    'after': before_metrics,
                    'action_succeeded': True,
                    'refusal': False,
                    'python_executed': True,
                    'result': str(result) if result is not None else None,
                }
            )

        except Exception as e:
            channels = self.measure_channels()
            return (
                channels,
                channels,
                {
                    'before': before_metrics,
                    'after': before_metrics,
                    'action_succeeded': False,
                    'refusal': False,
                    'python_error': str(e),
                    'error_type': type(e).__name__,
                }
            )

    def close(self):
        try:
            if hasattr(self, 'page'): self.page.close()
            if hasattr(self, 'context'): self.context.close()
            if hasattr(self, 'browser'): self.browser.close()
            if hasattr(self, 'playwright'): self.playwright.stop()
        except:
            pass

