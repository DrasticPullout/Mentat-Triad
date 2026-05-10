"""uii_geometry.py — UII v19.0 — substrate-shared types.

Substrate-side types used across the closure operators and the reality
adapters: SubstrateState (immutable per-iteration container holding
S/I/P/A operator instances), GroundingSpec (built at action commit and
passed to grounding adapters for blank filling), reality and agent
adapter ABCs, and the BASE_AFFORDANCES set defining which channel ids
are commit-eligible.

Relation to the math spine: SubstrateState's S/I/P/A scalar properties
project the operator state onto the math spine's coordinate frame.
These projections are observational only — the substrate does not read
its own scalar projections to drive behaviour. Math spine quantities
(Φ, ∇Φ, Hessian, vol_opt, C_local, C_global) live in uii_observer and
are computed external to the loop. The substrate contains no scoring;
coherence's commit decision is structural pattern-match against
trajectory direction, not optimization of a scalar quality measure.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from uii_operators import (
    CausalEdge,
    CoherenceOperator,
    CompressionOperator,
    DEFAULT_CHANNELS,
    OperatorConsistencyCheck,
    PredictionOperator,
    SensingChannel,
    SensingOperator,
    SelfModifyingOperator,
)


BASE_AFFORDANCES: Set[str] = {
    'navigate', 'click', 'fill', 'type', 'read',
    'scroll', 'observe', 'delay', 'evaluate',
    'query_agent',
    'python',
    'llm_query',
}

INTERFACE_COUPLED_SIGNALS: Set[str] = {
    'dom_depth', 'element_count', 'link_count', 'button_count',
    'input_count', 'scroll_position', 'viewport_height', 'dom_complexity',
}

POTENTIALLY_INVARIANT_SIGNALS: Set[str] = {
    'response_latency',
    'content_entropy',
    'surface_change_rate',
    'interaction_density',
}

SUBSTRATE_DIMS: List[str] = ['S', 'I', 'P', 'A']


class SubstrateState:
    """Per-iteration immutable container for the four state operators.

    Holds the S, I, P, A operator instances produced by one closure
    iteration. Replaced atomically each iterate() call. The S/I/P/A
    scalar properties project each operator onto the math spine's
    coordinate frame for external observation tooling; the substrate
    itself does not read these projections.
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

    @property
    def S(self) -> float:
        return self.sensing.to_scalar_proxy()

    @property
    def I(self) -> float:
        return self.compression.to_scalar_proxy()

    @property
    def P(self) -> float:
        return self.prediction.to_grounded_proxy()

    @property
    def A(self) -> float:
        return self.coherence.to_scalar_proxy()

    def as_dict(self) -> Dict[str, float]:
        return {"S": self.S, "I": self.I, "P": self.P, "A": self.A}


class StateTrace:
    """Substrate-side stub for c_global readout.

    The full C_local / C_global computation (gradient-vs-trajectory
    alignment) lives in uii_observer.StateTrace and is computed
    externally against the substrate's emission. This stub exists so
    that reality adapters and a few logging paths that reference
    c_global in formatted output have an attribute to read; the
    substrate itself never reads gradient-vs-trajectory alignment.
    c_global stays at 1.0 — the architecture's commitment is that
    instantiated systems are in-protocol by construction. Whether the
    loop is actually in the regime is what the external observer
    decides.
    """

    def __init__(self, max_length: int = 1000):
        self.c_global: float = 1.0


@dataclass
class GroundingSpec:
    """Specification built at action commit, passed to grounding adapters.

    Carries the natural gradient direction (top_gradient_channels,
    desired_delta) and the dark-channel set the grounding adapter is
    asked to perturb. Reality-adapter-internal — does not flow back
    into operator update logic.
    """
    desired_delta:          Dict[str, float]
    dark_channels:          List[str]
    top_gradient_channels:  List[str]
    current_url:            str
    page_title:             str
    nat_grad_magnitude:     float = 0.0

    def sensing_target_summary(self, max_channels: int = 5) -> str:
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
        sorted_by_magnitude = sorted(
            self.desired_delta.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True
        )[:max_channels]
        return ", ".join(f"{cid}={v:+.3f}" for cid, v in sorted_by_magnitude)


SELF_AFFORDANCES: Set[str] = {
    'read_ledger',
    'write',
    'write_outreach',
}

RELATION_AFFORDANCES: Set[str] = {
    'ground_triplet', 'revise_triplet',
    'promote_basin', 'demote_basin',
}

ENVIRONMENT_AFFORDANCES: Set[str] = {
    'navigate', 'click', 'fill', 'type', 'read', 'scroll',
    'observe', 'delay', 'evaluate', 'python',
    'write_file', 'send_email', 'query_agent',
}


class AgentHandler(ABC):
    """ABC for an external agent (LLM or peer) the substrate may query."""

    @abstractmethod
    def post_query(self, triad_id: str, query_text: str): ...

    @abstractmethod
    def get_response(self, triad_id: str) -> Optional[str]: ...


class UserAgentHandler(AgentHandler):
    """Console-mediated agent handler. Queries print to stdout and
    responses arrive via respond()."""

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
        print(f"Or leave pending — Triad will continue exploration")
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


class RealityAdapter(ABC):
    """ABC for a reality adapter: probe() returns channel-signal dict;
    execute() commits an action and returns (pre_channels, post_channels,
    context); get_current_affordances() reports what's currently
    executable."""

    @abstractmethod
    def execute(self, action: Dict) -> Tuple[Dict[str, float], Dict]: ...

    @abstractmethod
    def execute_trajectory(self, trajectory: List[Dict]) -> Tuple[List[Dict], bool]: ...

    @abstractmethod
    def get_current_affordances(self) -> Dict: ...

    @abstractmethod
    def close(self): ...
