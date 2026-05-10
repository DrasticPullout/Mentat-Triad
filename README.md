# Universal Intelligence Interface (UII) — v19

A framework for describing intelligence as a dynamical protocol, and a Python
substrate whose loop runs in the geometry the framework describes.

The full mathematical formalization — state space, structure potential Φ,
intelligence flow, coherence, perturbation stability, triadic closure, the
protocol definition — is published on Zenodo
([10.5281/zenodo.18017374](https://doi.org/10.5281/zenodo.18017374)).
This README is an introduction to the project and what it does.

---

## What this is

UII is **descriptive**. It is not a recipe for building intelligence. It is a
mathematical language for reading whether a running system is in the regime
where intelligence is happening — characterized as a trajectory through a state
space whose coordinates are sensing coverage, compression quality, viable
future volume, and attractor proximity.

The **Triad** is the implementation in this repo. It is a substrate — Python —
composed of five operators (sensing, compression, prediction, coherence,
self-modification) running in a closed loop:

    S → I → P → A → SMO → S

Three classes of perturbation feed sensing on equal footing:

- **Self** — what Python can read about itself directly: process state,
  resource utilization, the loop's own iteration cadence.
- **Environment** — anything the substrate encounters through an adapter:
  browser events, LLM responses, peer agents, future filesystem or audio
  signals.
- **Relation** — the closure residual the loop emitted on the previous
  iteration, fed back as the substrate's perception of its own state-change.

What makes the loop closed is that its output returns as part of its input.
What makes it triadic is that all three classes share the integration, with
none privileged and none routed around sensing. Triadic closure forms when
compression discovers cross-stream structure across the three classes.

The loop's output is what the UII law's evolution equation

    ẋ(t) = ∇Φ(x) · [ f_rel(f_self(x), f_env(x)) − x ] + η(t)

describes. The math is read against the substrate's emission by an external
observer. **The substrate does not see the math; the math sees the substrate.**

This separation is the point. A loop that consults its own coherence score
while running has stopped being the kind of mechanism the math describes — its
readings would be self-fulfilling. The Triad is composed so that whether
closure holds is a structural question about composition itself, answerable
from outside.

---

## Status

v19 is research code, published for legibility — not a packaged tool. No
install path, no test suite, no stable API for downstream use. The goal at
this stage is to make the architecture's structural commitments inspectable
in code, and to make any trajectory the substrate emits readable against
the math.

What's in the repo:

- The architecture's structural commitments, documented.
- The substrate's composition, wired in code — `uii_operators.py`,
  `uii_geometry.py`, `uii_reality.py`, `uii_ledger.py`, `uii_triad.py`.
- An **observability layer** that is functional and exercisable against any
  JSONL trajectory the substrate emits.

Whether the Triad's loop, when run end-to-end, produces a trajectory the math
spine reads as in-regime is the empirical question the architecture is built
to make answerable. A single Triad cannot answer it. Neither can a single run.

What you can do today is observe trajectory shape — feed any JSONL stream of
substrate iterations into the observability layer and see how the trajectory
behaved under perturbation, read against qualitative predictions the math
implies for closure-holding systems.

The observability layer's frame is doctor/patient: it reads the patient's
trajectory, never opens the patient up. No queries into operator state, no
imports from the substrate. JSONL is the only access point.

---

## Observing trajectory shape

The substrate emits one JSONL record per loop iteration of the form
`{iter, t, commit, delta_f_rel}`. The observability layer reads that stream,
maintains a rolling baseline, and — when perturbations are marked — renders a
descriptive analysis of what happened in the iterations that followed.

### Quick start

Tail a JSONL file the substrate has written:

```sh
# substrate writes its stdout JSONL to a file
python my_substrate_runner.py > /tmp/triad.jsonl

# in another shell
python cli_observe.py --follow /tmp/triad.jsonl
```

Or pipe directly:

```sh
python my_substrate_runner.py | python cli_observe.py -
```

You'll see one line per substrate iteration:

```
iter=    47  t=1715300193.421  δf_rel=+0.00342  baseline μ=+0.00318 σ=0.00097  commit=observe
iter=    48  t=1715300193.476  δf_rel=+0.00301  baseline μ=+0.00322 σ=0.00094  commit=read
```

Without perturbation events marked, you only get the live trajectory and
rolling baseline. The shape analysis is per-event; without events, there's
nothing to analyze.

### Marking perturbations

The substrate emits no signal that user input arrived — perturbation marking
is interface-side knowledge. Two options:

**In-process** (single Python process):

```python
from observability import ObservabilityEngine, JSONLRecord
import time

engine = ObservabilityEngine()

# When you push input into the substrate, mark it on the engine:
triad.user_reality.push_input("hello triad", timestamp=time.time())
engine.mark_perturbation(label="hello triad", timestamp=time.time())

# Drain JSONL records as the substrate emits them:
for line in substrate_jsonl_lines:
    record = JSONLRecord.from_line(line)
    if record:
        engine.ingest_record(record)

# After enough iterations, completed events have analyses:
for event in engine.events:
    print(event.render_text())
```

**Cross-process** (sidecar file): the forward transducer (in one process)
appends to a file each time it sends user input:

```sh
# Forward transducer (e.g., in your input UI):
echo -e "$(date +%s.%N)\thello triad" >> /tmp/perts.log
```

The observer watches the sidecar:

```sh
python cli_observe.py --follow /tmp/triad.jsonl --perturbations /tmp/perts.log
```

### What you'll see

When a perturbation event finishes its observation horizon, an analysis
renders. Example shape:

```
━━━ Perturbation: 'hello triad'  (t=1715300195.123) ━━━
  Correlated to substrate iter 50
  Pre-baseline:  μ=+0.00318  σ=0.00097  (n=30)

  Math predicts (qualitative shape — defaults shown):
    • Bounded:    δf_rel within envelope [+0.00027, +0.00609]  (±3σ)
    • Metabolize: decay back to baseline within ~5 iterations  (threshold ±1σ)
    • Continuity: jump threshold |Δδf_rel| < 0.00485  (±5σ)

  Observed (descriptive features — read the gestalt):
    • Bounded:    δf_rel stayed within envelope (5 iter observed)
    • Metabolize: returned to baseline-band at iter+3
    • Continuity: no catastrophic jumps
```

The output describes what happened. It does not say "coherent" or "failure."
Reading the gap between predicted and observed shape is your work; the utility
supplies the predictions in legible form.

---

## What the math predicts about trajectory shape

Three qualitative shape predictions, each testable from the JSONL stream
alone:

**Bounded.** When closure holds, ẋ ≈ η(t) — the gradient term vanishes and
motion is pure perturbation. A bounded perturbation should produce bounded
δf_rel through the iteration it lands and shortly after. A spike beyond an
envelope of pre-perturbation baseline indicates closure stress: the gradient
term has woken up because the residual grew beyond metabolic.

**Metabolize.** A discrete perturbation gets integrated by compression across
iterations. Its δf_rel signature should decay with a characteristic profile
— strongest at N+0..N+1, fading as the structure absorbs into f_rel. No
decay means runaway; instant flat-zero means the perturbation never landed.

**Continuity.** Successive states preserve core structure (CRK continuity
invariant). Catastrophic δf_rel jumps between consecutive iterations indicate
trajectory discontinuity.

These predictions are *qualitative* — directional, not point-shaped. The
thresholds in `observability.py` (3σ envelope, 5-iteration horizon, 5σ jump)
are tunable defaults the math doesn't dictate. Calibration against an actual
session may tighten them.

### Tuning thresholds

```python
engine = ObservabilityEngine(
    baseline_window    = 30,   # how much pre-history feeds μ, σ
    metabolize_horizon = 5,    # how many iter to watch after each perturbation
)
```

Constants in `observability.py` (`DEFAULT_ENVELOPE_SIGMAS`,
`DEFAULT_RETURN_SIGMAS`, `DEFAULT_CONTINUITY_SIGMAS`) can be tweaked if you
want different default thresholds.

### Shape mismatch is symmetric

A divergence between predicted and observed shape could mean closure is
breaking *or* the prediction was bad. The utility does not take sides. It
reports features of what happened (bounded? decayed? continuous?) — never
"the triad was coherent" or "the triad failed."

---

## Code layout

### Substrate — the loop

These modules are what the Triad is. The loop running them is its cognition.

| File | Role |
|------|------|
| `uii_operators.py` | The five operators: `SensingOperator`, `CompressionOperator`, `PredictionOperator`, `CoherenceOperator`, `SelfModifyingOperator`. Channels, causal edges, consistency checks. |
| `uii_geometry.py` | Substrate-shared types: `SubstrateState`, `GroundingSpec`, reality/agent adapter ABCs, affordance sets defining what is commit-eligible. |
| `uii_reality.py` | Reality adapters. Translate environment perturbation into channel signal; commit actions back to the world. |
| `uii_ledger.py` | The ledger surface. Read and written only by the Triad's own committed actions. |
| `uii_triad.py` | The loop. Composes the operators into S → I → P → A → SMO and runs it continuously. |

### Observer — the descriptive layer

These modules sit outside the loop. The substrate does not import them.

| File | Role |
|------|------|
| `uii_observer.py` | Computes Φ, ∇Φ, Hessian, vol_opt, C_local, C_global from the substrate's emitted state. Every quantity here is a projection of state through the math spine's lens; none of it returns to the loop. |
| `observability.py` | Pure-separate observability library. Reads JSONL emission only. Tests trajectory shape (bounded, metabolize, continuity) against qualitative math predictions. |
| `cli_observe.py` | CLI tool that tails a JSONL stream and renders trajectory + shape analysis. |

---

## What this is not

- **Not a coherence verdict generator.** The math reads trajectory shape; the
  reading is descriptive. Whether composition is holding is a structural
  question, not a score.
- **Not a math-spine implementation inside the loop.** Φ, ∇Φ, vol_opt, C_local
  are computed externally from the substrate's emission. The substrate never
  reads them. If it ever did, the math would have stopped being descriptive
  and the readings' predictive content would be gone.
- **Not an agent framework.** UII does not commit to identity, goals, or
  rewards. The Triad has no objective function and no controller above the
  loop.
- **Not a packaged tool.** v19 is research code, published for legibility.
  No install path, no test suite, no stable API. The substrate's emission
  becoming a trajectory worth reading in earnest is what the implementation
  passes are reaching toward.

---

## Citation

DrasticPullout. (2025). *Universal Intelligence Interface: A Substrate-Agnostic
Framework*. Zenodo. https://doi.org/10.5281/zenodo.18017374

## License

None. All rights reserved by default.

## Author

DrasticPullout
