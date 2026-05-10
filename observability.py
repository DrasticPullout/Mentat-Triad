"""observability.py — UII v19.0 — pure observability library.

Reads the substrate's JSONL emission stream and applies math-spine shape
predictions to the trajectory. Imports nothing from the substrate. The
substrate has no awareness this exists.

Architectural commitments enforced by structure:
  - JSONL is the only access point. No queries into operator state.
  - The substrate's emission is the public surface; what we get is
    what the loop produces.
  - Math-spine quantities (Φ, ∇Φ, vol_opt) are NOT computed here —
    they require operator-internal state we don't have. What this
    library does instead is test trajectory SHAPE against qualitative
    predictions the math implies for closure-holding systems under
    perturbation.
  - Shape predictions are descriptive ranges, not point predictions.
    Math gives us shape, not specifics. Quantitative thresholds here
    are tunable defaults; the interface should communicate this.
  - Shape-mismatch is symmetric — a divergence between predicted and
    observed shape could mean closure is breaking OR the user's
    prediction was bad. This library reports descriptive features;
    interpretation is the user's work.

What the math predicts about trajectory shape under perturbation,
testable from {iter, t, commit, delta_f_rel} alone:

  Closure-holding bound. ẋ ≈ η(t) when closure holds. Bounded
  perturbation → bounded δf_rel through the iteration it lands and
  shortly after. A spike means closure is breaking — gradient term
  has woken up because residual grew beyond metabolic.

  Metabolization profile. A discrete perturbation gets integrated by
  compression across iterations. Its δf_rel signature should decay
  with a characteristic shape — strongest at N+0..N+1, fading as
  structure absorbs into f_rel. No decay means runaway; instant flat
  means it never landed.

  Continuity preservation. Successive states preserve core structure.
  Catastrophic δf_rel jumps indicate closure stress.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Tunable qualitative-shape defaults.
#
# These are the math's qualitative predictions made specific enough to
# test. They are NOT derived from the math alone — the math says
# "bounded," "decays," "continuous"; these constants pick concrete
# thresholds. Calibration against an actual session may tighten them.
# The interface should communicate that these are defaults, not truth.
# ─────────────────────────────────────────────────────────────────────
DEFAULT_BASELINE_WINDOW    = 30    # iterations of pre-perturbation history
DEFAULT_ENVELOPE_SIGMAS    = 3.0   # bounded-ness threshold (× baseline σ)
DEFAULT_METABOLIZE_HORIZON = 5     # iterations to expect decay-to-baseline
DEFAULT_RETURN_SIGMAS      = 1.0   # "returned to baseline" threshold (× σ)
DEFAULT_CONTINUITY_SIGMAS  = 5.0   # catastrophic-jump threshold (× σ)


# ─────────────────────────────────────────────────────────────────────
# Record types
# ─────────────────────────────────────────────────────────────────────
@dataclass
class JSONLRecord:
    """One iteration of the substrate's emission. Schema = v18.9."""
    iter:        int
    t:           float
    commit:      Optional[Dict[str, Any]]
    delta_f_rel: float

    @classmethod
    def from_line(cls, line: str) -> Optional['JSONLRecord']:
        try:
            d = json.loads(line)
            return cls(
                iter        = int(d['iter']),
                t           = float(d['t']),
                commit      = d.get('commit'),
                delta_f_rel = float(d['delta_f_rel']),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None


# ─────────────────────────────────────────────────────────────────────
# Rolling baseline
# ─────────────────────────────────────────────────────────────────────
class Baseline:
    """Rolling window of δf_rel values. Provides mean, std, envelope.

    The envelope is the math's "bounded" prediction made concrete at
    a chosen sigma threshold. Default 3σ is a qualitative pick, not
    derivation from the math.
    """

    def __init__(self, window: int = DEFAULT_BASELINE_WINDOW):
        self.window = window
        self._values: deque = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._values.append(value)

    @property
    def n(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        return statistics.mean(self._values) if self._values else 0.0

    @property
    def std(self) -> float:
        # statistics.stdev needs ≥2 points; below that, return 0
        return statistics.stdev(self._values) if len(self._values) >= 2 else 0.0

    def envelope(self, sigmas: float = DEFAULT_ENVELOPE_SIGMAS) -> Tuple[float, float]:
        m, s = self.mean, self.std
        return (m - sigmas * s, m + sigmas * s)

    def snapshot(self) -> 'Baseline':
        """Frozen copy of current baseline state. Used to capture
        pre-perturbation conditions for a PerturbationEvent."""
        copy = Baseline(window=self.window)
        copy._values = deque(self._values, maxlen=self.window)
        return copy


# ─────────────────────────────────────────────────────────────────────
# Perturbation event
# ─────────────────────────────────────────────────────────────────────
@dataclass
class PerturbationEvent:
    """A user perturbation, with its predicted and observed shape.

    The forward transducer (interface side) calls
    ObservabilityEngine.mark_perturbation() at the moment it pushes
    user input into the substrate. Subsequent JSONL records are
    correlated to the event via timestamp; at the moment of
    correlation, the engine snapshots the baseline as
    pre_baseline. Subsequent records up to metabolize_horizon are
    attached to observed_trajectory.

    Snapshot at correlation time (not at mark time) is what allows
    both in-process integration (mark is near-simultaneous with the
    correlated iteration) and CLI sidecar mode (events loaded up-front
    before JSONL is processed) to produce correct pre-baseline reads.
    """
    timestamp:            float
    label:                str
    pre_baseline:         Optional[Baseline]   = None      # filled at correlation
    iter_at_perturbation: Optional[int]        = None
    observed_trajectory:  List[JSONLRecord]    = field(default_factory=list)
    metabolize_horizon:   int                  = DEFAULT_METABOLIZE_HORIZON

    # ─────── Math-derived qualitative predictions ───────
    def predict_shape(self) -> Dict[str, Any]:
        """Qualitative shape predictions implied by the math.

        Each prediction has a 'note' explaining what the math says,
        and concrete thresholds chosen as qualitative defaults. The
        thresholds are tunable; the math's predictions are
        directional, not point-shaped.

        If pre_baseline is not yet filled (event not yet correlated),
        returns a status indicating insufficient context.
        """
        if self.pre_baseline is None:
            return {'status': 'baseline not yet captured (event not correlated)'}

        env = self.pre_baseline.envelope(DEFAULT_ENVELOPE_SIGMAS)
        return {
            'bounded': {
                'envelope':           env,
                'envelope_sigmas':    DEFAULT_ENVELOPE_SIGMAS,
                'note': ('closure-holding limit ẋ ≈ η(t): bounded perturbation '
                         '→ bounded δf_rel; spike beyond envelope indicates '
                         'closure stress'),
            },
            'metabolize': {
                'horizon_iterations': self.metabolize_horizon,
                'return_threshold':   DEFAULT_RETURN_SIGMAS * self.pre_baseline.std,
                'return_sigmas':      DEFAULT_RETURN_SIGMAS,
                'note': ('compression integrates the perturbation across '
                         'iterations; effect should decay toward baseline '
                         'within ~horizon iterations'),
            },
            'continuity': {
                'jump_threshold':     DEFAULT_CONTINUITY_SIGMAS * self.pre_baseline.std,
                'jump_sigmas':        DEFAULT_CONTINUITY_SIGMAS,
                'note': ('successive states preserve core structure; '
                         'catastrophic δf_rel jumps indicate trajectory '
                         'discontinuity'),
            },
        }

    # ─────── Descriptive observation reads ───────
    def observe_shape(self) -> Dict[str, Any]:
        """Descriptive structural reads from observed trajectory.

        These are FEATURES of what happened, not verdicts. The user
        reads the gestalt and forms judgment. Interface does NOT
        decide whether the triad was 'coherent.'
        """
        if self.pre_baseline is None:
            return {'status': 'baseline not yet captured (event not correlated)'}
        if not self.observed_trajectory:
            return {'status': 'no trajectory observed yet'}

        deltas = [r.delta_f_rel for r in self.observed_trajectory]
        env_low, env_high = self.pre_baseline.envelope(DEFAULT_ENVELOPE_SIGMAS)

        # Bounded-ness: max excursion outside envelope (0.0 if always inside)
        max_excursion = 0.0
        excursion_iter_offset = None
        for i, d in enumerate(deltas):
            if d > env_high:
                e = d - env_high
                if e > max_excursion:
                    max_excursion = e
                    excursion_iter_offset = i
            elif d < env_low:
                e = env_low - d
                if e > max_excursion:
                    max_excursion = e
                    excursion_iter_offset = i

        # Metabolization: when did δf_rel return to within return_threshold of baseline mean?
        return_threshold = DEFAULT_RETURN_SIGMAS * self.pre_baseline.std
        target_mean      = self.pre_baseline.mean
        return_iter      = None
        for i, d in enumerate(deltas):
            if abs(d - target_mean) <= return_threshold:
                return_iter = i
                break

        # Continuity: rate-of-change exceedances between consecutive iterations.
        # Within a metabolization window these are typically the perturbation
        # arriving and decaying — they're characteristic of the metabolization
        # profile, not necessarily closure stress. The user reads in context.
        jump_threshold = DEFAULT_CONTINUITY_SIGMAS * self.pre_baseline.std
        exceedances: List[Dict[str, Any]] = []
        for i in range(1, len(deltas)):
            jump = abs(deltas[i] - deltas[i - 1])
            if jump > jump_threshold and jump_threshold > 0:
                exceedances.append({
                    'iter_offset':    i,
                    'jump_magnitude': jump,
                    'threshold':      jump_threshold,
                })

        return {
            'status': 'observed',
            'iterations_observed': len(deltas),
            'bounded': {
                'stayed_within_envelope':  max_excursion == 0.0,
                'envelope_max_excursion':  max_excursion,
                'excursion_iter_offset':   excursion_iter_offset,
            },
            'metabolize': {
                'returned_to_baseline':    return_iter is not None,
                'return_iter_offset':      return_iter,
            },
            'continuity': {
                'rate_of_change_exceedances':       exceedances,
                'rate_of_change_exceedance_count':  len(exceedances),
            },
        }

    # ─────── Legible rendering — descriptive only, never verdictal ───────
    def render_text(self) -> str:
        prediction  = self.predict_shape()
        observation = self.observe_shape()

        lines: List[str] = []
        lines.append(f"━━━ Perturbation: {self.label!r}  (t={self.timestamp:.3f}) ━━━")

        if self.iter_at_perturbation is not None:
            lines.append(f"  Correlated to substrate iter {self.iter_at_perturbation}")
        else:
            lines.append(f"  Not yet correlated to a substrate iteration "
                         f"(no baseline captured)")
            return '\n'.join(lines)

        # By this point pre_baseline is set (filled at correlation)
        lines.append(f"  Pre-baseline:  μ={self.pre_baseline.mean:+.5f}  "
                     f"σ={self.pre_baseline.std:.5f}  "
                     f"(n={self.pre_baseline.n})")
        lines.append("")

        # Predictions
        lines.append("  Math predicts (qualitative shape — defaults shown):")
        b = prediction['bounded']
        lines.append(f"    • Bounded:    δf_rel within envelope "
                     f"[{b['envelope'][0]:+.5f}, {b['envelope'][1]:+.5f}]  "
                     f"(±{b['envelope_sigmas']:.0f}σ)")
        m = prediction['metabolize']
        lines.append(f"    • Metabolize: decay back to baseline within "
                     f"~{m['horizon_iterations']} iterations  "
                     f"(threshold ±{m['return_sigmas']:.0f}σ)")
        c = prediction['continuity']
        lines.append(f"    • Continuity: jump threshold "
                     f"|Δδf_rel| < {c['jump_threshold']:.5f}  "
                     f"(±{c['jump_sigmas']:.0f}σ)")
        lines.append("")

        # Observations — descriptive only
        lines.append("  Observed (descriptive features — read the gestalt):")
        if observation.get('status') != 'observed':
            lines.append(f"    {observation.get('status', 'unknown')}")
            return '\n'.join(lines)

        b = observation['bounded']
        if b['stayed_within_envelope']:
            lines.append(f"    • Bounded:    δf_rel stayed within envelope "
                         f"({observation['iterations_observed']} iter observed)")
        else:
            lines.append(f"    • Bounded:    max excursion outside envelope = "
                         f"{b['envelope_max_excursion']:.5f} "
                         f"at iter+{b['excursion_iter_offset']}")

        m = observation['metabolize']
        if m['returned_to_baseline']:
            lines.append(f"    • Metabolize: returned to baseline-band "
                         f"at iter+{m['return_iter_offset']}")
        else:
            lines.append(f"    • Metabolize: not yet returned to baseline-band "
                         f"(observed {observation['iterations_observed']} iter)")

        c = observation['continuity']
        if c['rate_of_change_exceedance_count'] == 0:
            lines.append(f"    • Continuity: no rate-of-change exceedances "
                         f"above threshold")
        else:
            lines.append(f"    • Continuity: {c['rate_of_change_exceedance_count']} "
                         f"rate-of-change exceedance(s) above threshold "
                         f"— within metabolization, these are typically the "
                         f"perturbation's entry/decay signature")
            for j in c['rate_of_change_exceedances'][:3]:
                lines.append(f"        - iter+{j['iter_offset']}  "
                             f"|Δ| = {j['jump_magnitude']:.5f}")

        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────
class ObservabilityEngine:
    """The reading-the-trajectory utility. Reads JSONL records, maintains
    rolling baseline, correlates user-perturbation events to the
    trajectory, and produces shape-prediction reads.

    No substrate state is queried. The substrate is structurally
    invisible to this engine except through the JSONL stream.

    Forward-transducer integration:
      When the interface pushes user input to the substrate, it should
      also call engine.mark_perturbation(label) on this engine. The
      engine snapshots the current baseline and watches subsequent
      JSONL records for the trajectory shape.

      This timestamp correlation is interface-side knowledge ('I just
      sent input at time T') — the substrate emits no signal that
      input arrived. Two facts, correlated outside the substrate.
    """

    def __init__(self,
                 baseline_window:    int = DEFAULT_BASELINE_WINDOW,
                 metabolize_horizon: int = DEFAULT_METABOLIZE_HORIZON):
        self.baseline           = Baseline(window=baseline_window)
        self.metabolize_horizon = metabolize_horizon
        self.records:          List[JSONLRecord]      = []
        self.events:           List[PerturbationEvent] = []  # completed events
        self._pending_events:  List[PerturbationEvent] = []  # awaiting trajectory

    # ───── ingest ─────
    def ingest_record(self, record: JSONLRecord) -> None:
        """Process one JSONL record. Updates baseline, attaches the
        record to any pending events whose horizon includes this iter,
        and migrates events to the completed list when their horizon
        is reached."""
        self.records.append(record)

        record_in_event_window = False

        # First pass: correlate any uncorrelated events to this iteration
        # if their timestamp is at-or-before this record's timestamp.
        # At correlation, snapshot the current baseline as pre_baseline —
        # this makes the predictions reflect baseline as of the
        # perturbation moment, not as of mark-time.
        for event in self._pending_events:
            if event.iter_at_perturbation is None and record.t >= event.timestamp:
                event.iter_at_perturbation = record.iter
                event.pre_baseline         = self.baseline.snapshot()

        # Second pass: attach record to events within their horizon,
        # finalize events that have reached their horizon.
        for event in list(self._pending_events):
            if event.iter_at_perturbation is None:
                continue
            offset = record.iter - event.iter_at_perturbation
            if 0 <= offset < event.metabolize_horizon:
                event.observed_trajectory.append(record)
                record_in_event_window = True
            elif offset >= event.metabolize_horizon:
                # Move to completed
                self.events.append(event)
                self._pending_events.remove(event)

        # Update baseline only with non-event records (events shouldn't
        # contaminate the baseline used for predicting other events).
        if not record_in_event_window:
            self.baseline.update(record.delta_f_rel)

    def mark_perturbation(self,
                          label:     str,
                          timestamp: Optional[float] = None) -> PerturbationEvent:
        """Called by the forward transducer at the moment it pushes a
        user input into the substrate. Baseline snapshot is deferred
        until the engine correlates the event to a substrate iteration
        (in ingest_record) — this ensures pre_baseline reflects the
        baseline AT the perturbation moment, regardless of when
        mark_perturbation was called relative to record ingestion."""
        ts = timestamp if timestamp is not None else time.time()
        event = PerturbationEvent(
            timestamp          = ts,
            label              = label,
            pre_baseline       = None,    # filled at correlation
            metabolize_horizon = self.metabolize_horizon,
        )
        self._pending_events.append(event)
        return event

    # ───── streams ─────
    def stream(self, lines: Iterable[str]) -> Iterator[JSONLRecord]:
        """Stream JSONL lines through the engine. Yields each parsed
        record as it's ingested, so callers can render live."""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            record = JSONLRecord.from_line(line)
            if record is None:
                continue
            self.ingest_record(record)
            yield record

    # ───── snapshots ─────
    def all_events(self) -> List[PerturbationEvent]:
        """Return all events — completed and pending — in mark order."""
        # pending events appear first because they were marked but
        # haven't completed yet; completed events come after in the
        # order they were finalized.
        return list(self._pending_events) + list(self.events)
