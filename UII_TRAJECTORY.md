# UII Development Trajectory

> Quick Reference For LLMs

**Current version:** `v19.0`
**Phase 14:** Interface Era *(working name — phase newly entered)*

**Citation:** DrasticPullout. (2025). *Intelligence as a Universal Protocol.* Zenodo. <https://doi.org/10.5281/zenodo.18017374>

---

## Table of Contents

- [Core Invariants (Never Change)](#core-invariants-never-change)
- [Mathematical Foundation](#mathematical-foundation)
- [DASS Substrate](#dass-substrate)
- [CRK Constraints](#crk-constraints)
- [Evolution Arc](#evolution-arc)
  - [Phases 1–8: Foundation through Mortality/Replication](#phases-18-foundation-through-mortalityreplication-v1v138)
  - [Phase 9: Living Causal Model Era (v14.x)](#phase-9-living-causal-model-era-v14x)
  - [Phase 10: Structural Inference Era (v15.x)](#phase-10-structural-inference-era-v15x)
  - [Phase 11: Geometric Dynamics Era (v16.x)](#phase-11-geometric-dynamics-era-v16x)
  - [Phase 12: Continuous DASS Loop Era (v17.x)](#phase-12-continuous-dass-loop-era-v17x)
  - [Phase 13: Multi-Agent Field Era (v18.x — landed at v18.9)](#phase-13-multi-agent-field-era-v18x--landed-at-v189)
  - [Phase 14: Interface Era (v19.x — beginning)](#phase-14-interface-era-v19x--beginning)
- [iterate() Architecture](#iterate-architecture-v189v190--strict-io-closure-loop-with-jsonl-emission)
- [Migration Trigger](#migration-trigger-still-fully-stripped)
- [Key Components](#key-components-v190)
- [Eliminated — Do Not Re-Introduce](#eliminated--do-not-re-introduce)
- [Affordance Sets](#affordance-sets-v190)
- [File Inventory](#file-inventory-v190--nine-files)
- [Ledger Schema](#ledger-schema-v190)
- [Architecture Constants](#architecture-constants-v190)
- [Validation Signals](#validation-signals-v190--healthy-run)
- [Warning Signals (Drift)](#warning-signals-v190-drift--architectural-backsliding)
- [Deprecated / Removed Concepts](#deprecated--removed-concepts)
- [Canonical Reminders](#canonical-reminders-v190)
- [Evolution Cycle](#evolution-cycle)
- [Trajectory Open Issues](#trajectory-open-issues-carry-into-v191)
- [Version Control Test](#version-control-test)

---

## Core Invariants (Never Change)

- Intelligence = protocol, not agent/goal/trait
- Substrate-agnostic (any causal medium)
- No externally imposed optimization targets, rewards, or identity
- Optionality preservation is a dominant criterion
- Coherence loss = protocol exit (by design — descriptive, not enforced via `raise`; v18.3+)
- Recursive `Input → Operator → Output` composition, with **Triadic Closure** as the base case of the recursion (named explicitly v18.8; implicit from Phase 1).

Every adaptive operation in the architecture has the shape `Input → Operator → Output`. Compositions of operations have the same shape. The recursion terminates in Triadic Closure — the structural circuit between sensing, compression, prediction, coherence, and self-modification, where outputs feed back into inputs through a single typed pathway without external orchestration. Below that level you have computation. At that level you have the smallest adaptive system. Above that level you have composite adaptive systems built from triadic-closure blocks under the same `I → O → O` pattern. The structure is fractal: the whole `iterate()` loop has the same shape as a single operator's `apply()`.

**v18.8 addition:** the substrate contains no scoring inside it. The math spine is a description of the trajectory composition produces, computed externally against the substrate's emitted state. Any path that would have the substrate read its own scalar quality measure and adjust behaviour is incoherent with the architecture.

**v18.9 addition:** the substrate's only output is its causality. Each iteration emits one JSONL record on stdout: `{iter, t, commit, delta_f_rel}` — `commit` is the cause the substrate pushed into the world, `delta_f_rel` is the per-channel motion that resulted. That's `{input, ẋ}` in the math spine's coordinate frame. Nothing else is permitted as emission content: no scoring (forbidden by [Canonical Reminder #1](#canonical-reminders-v190)), no math-spine readings (those live in `uii_observer`, external to the substrate), no self-assessment, no ceremonial events. The schema isn't a v18.9 design choice; it's what's left after the substrate is correctly closed.

**v19.0 addition:** the substrate is interface-bound. Two surfaces: forward transducers route external perturbation into the existing `env_signal` pathway (discontinuous input, on the perturber's schedule); pure JSONL consumers read the trajectory and report shape (bounded / metabolize / continuity) without ever touching operator state. Phase 13 dissolved the LLM/substrate boundary; Phase 14 establishes explicit interface surfaces around the substrate.

### The trio of load-bearing invariants

Coherence is what holds the loop together as a process — without trajectory alignment to ∇Φ, the loop is no longer a coherent computation. Optionality is what the loop preserves over time — without continued reachability, future state space contracts. Recursive `Input → Operator → Output` with Triadic Closure as base case is how any adaptive computation in this architecture is constructed — the structural shape that all three levels (single operator, full closure, composite system) share.

Every version since Phase 1 has been an attempt to make triadic closure real, but the third invariant became namable only in v18.8, when the closure was structural for the first time. The three compose: closure with strict typed I/O produces coherent trajectories, and coherent trajectories preserve optionality.

`v18.8` is the base version of an adaptive system in this architecture — the recursion finally has a real base case to terminate in. `v18.9` makes that base emit cleanly; `v19.0` lets it be perturbed and observed from outside. Future work composes upward from this base.

---

## Mathematical Foundation

### The Compressed UII Equation (Mathematical Spine)

```
ẋ = P_M(G⁻¹(x) · ∇Φ(x))

G(x) = H(x) = Hessian(Φ)(x)      information metric
M    = {x : T(x) = 0}            triadic constraint manifold
P_M                              projection onto M

Lyapunov: dΦ/dt = ∇Φᵀ · G⁻¹ · ∇Φ ≥ 0
Guaranteed when H positive semi-definite. Φ non-decreasing within M.
```

### Full three-term potential (Mathematical Spine §2)

```
Φ(x) = α·C(x) + β·log(O(x)) + γ·K(x)

C(x) = Σ_edges |w(e)|·conf(e)·cov(src)·cov(tgt)   normalized to [0,1]
O(x) = Σ of positive eigenvalues of Σ_P            (prediction covariance)
K(x) = -‖coverage_current - coverage_peak‖²        (proximity to attractor)

H = α·H_C + β·H_O + γ·H_K + ε·I

∇Φ[ch_i] = Σ_{e:src=i} w·conf·cov(tgt)
         + Σ_{e:tgt=i} w·conf·cov(src)
         normalized to unit vector
```

### Loop-rate proxies and accounting

```
C_local  = ⟨∇Φ, ẋ⟩ / (‖∇Φ‖ · ‖ẋ‖)
C_global = ∫ C_local dt           (primary run quality measure;
                                   initialized = 1.0 per Spine §11)
δ²Φ = 0.5 · δxᵀ · H · δx         (Lyapunov accounting)
```

**v18.8 placement:** Φ, ∇Φ, Hessian, Σ_P, vol_opt, C_local, C_global all live in `uii_observer.py`, which is not imported by the substrate. They are computed against the substrate's emitted state by a reader external to the loop. The math spine is fully descriptive; the substrate runs without ever reading its own scalar projections.

**v18.9 placement:** `uii_observer` is no longer imported by `uii_triad` either. The math-spine module exists for external tooling only.

**v19.0 placement:** a *second* external reader joins `uii_observer` — `observability.py`, which reads only the JSONL emission `{iter, t, commit, delta_f_rel}` and tests qualitative trajectory shape against the math's predictions (bounded, metabolize, continuity). It computes no Φ, no ∇Φ, no vol_opt — those need operator-internal state. Two readers, both external, with different access surfaces: `uii_observer` reads operator state directly (when given access in the same process); `observability` reads only the public JSONL stream.

**The emission is ẋ.** v18.9's `delta_f_rel` is the substrate's per-channel motion this iteration — bounded δf_rel that SMO clipped, in the math spine's coordinate frame, projected onto active channels. Together with `commit` (the cause that produced part of that motion), the emission is `{input, ẋ}`. The math spine's central dynamical object — the trajectory ẋ = P_M(G⁻¹ · ∇Φ) — is now a public artifact. The math spine wasn't just describing what the substrate is; it was describing what the substrate would emit if it had an emission port. v18.9 built the port. v19.0's external readers read what was, mathematically, already the right thing to read.

### Action scoring removed (v18.8)

The diagnostic `nat_grad_align(a)` / `score(a)` formalism in earlier versions and the `rank_actions_by_eog` mechanism that drove commits through v18.7 are no longer present anywhere in the substrate. Coherence's commit gate is structural pattern-match against trajectory direction (see [iterate() Architecture](#iterate-architecture-v189v190--strict-io-closure-loop-with-jsonl-emission)). The substrate does not score affordances — it audits projections.

---

## DASS Substrate

```
S: Sensing      [0,1]
I: Integration  [0,1]
P: Prediction   [0,1]
A: Attractor    [0,1]   (optimal ~0.7)
```

In v15.1+ these are proper mathematical operators, not scalars. Each exposes `to_scalar_proxy()` for projection onto the math spine's coordinate frame.

**v18.8 reframing.** The four operators run under strict I/O at every boundary. Each operator's `apply()` takes a single positional argument that is the prior operator's output. `to_scalar_proxy()` and `to_grounded_proxy()` are preserved as projections for external observation — the substrate itself does not consume these scalars. Per-channel covariance for Σ_P is built directly by `uii_observer._build_sigma_p` from compression's `causal_graph` and sensing's coverage. Self-channels are now Python-introspection signals (`self/clock/...`, `self/process/...`, `self/gc/...`), not scalar reductions of operator state — sensing reads these on equal terms with environment probes and compression discovers cross-class edges between them.

---

## CRK Constraints

**Status in v18.8+:** not present in the substrate as an executable module. The `CRKMonitor` class, `observe()`, the C1–C7 firing logic, the load-bearing presence assertion (v18.5), and the SMO `repair_directives` pathway have all been removed from the codebase. The constraint set was originally a separate diagnostic that observed and fed SMO. In v18.8 the principles those constraints expressed become structural in operator design rather than enforced by an external auditor:

- **C1 (Continuity)** — compression integrates sensing across iterations; sudden discontinuity surfaces as residual variance and reduces edge confidence by sign-disagreement.
- **C2 (Optionality)** — falls out compositionally when integration is correct: as `f_rel` grows through accumulated residuals, projections affordances admit also grow. Reachable state space is `f_rel` itself.
- **C3 (Non-Internalization)** — strict I/O at every operator boundary forbids any operator from reaching back through the loop. SMO's `δf_rel` re-enters sensing through `env_signal` on equal terms with every other perturbation.
- **C4 (Reality)** — sensing's coverage decays toward 0 on absent signal; channels deactivate after `INACTIVITY_THRESHOLD`. Reality manifests through coverage shape, not through a separate uncertainty injector.
- **C5 (Attribution)** — affordance-class channels carry the just-used spike that lets compression discover action→outcome edges as ordinary causal sources; external pressure surfaces in `api_llm` coverage degradation, not in a separate reclassifier.
- **C6 (Agenthood)** — agents are sensed through `AgentRealityAdapter`'s latency / token / availability channels; the agent_registry records what coherence's `loop_closure` looked like at each commit involving an agent, rather than running a separate MAC-FCK trust update.
- **C7 (Coherence)** — coherence's `signature_deviation` tracks the EMA-deviation of `f_rel` structural metrics (active_channels, graph_edges, mean_confidence). When `signature_deviation` exceeds `DEVIATION_THRESHOLD = 0.5`, `commit_decision` returns `None` — the substrate holds short while in flux.

The shift is from external auditor + repair_directives + SMO patching to principles built into operator structure. Coherence's `commit_decision` is the only gate, and it is structural pattern-match — not a scalar comparison.

---

## Evolution Arc

### Phases 1–8: Foundation through Mortality/Replication (v1–v13.8)

- **v1–v3:** DASS + CRK foundation, triadic closure, pure state architecture
- **v4–v6:** Real API, effectors, prediction error, visualization
- **v7–v8.2:** API evolution (Gemini → Ollama)
- **v9.x:** Basin discovery, metabolic decay → invariant perturbation
- **v10.x:** Action-flow architecture, Truth Verification Layer, optionality-driven basins
- **v11.x:** Mentat Triad, CNS signal hierarchy, minimal architecture, immediate logging
- **v12.x:** Continuous reality perturbation, normalized measurements, ENO-EGD, LatentDeathClock
- **v13.x:** Dual-budget mortality, AttractorMonitor, TriadGenome, CNSMitosisOperator, FAO meta-evolution, S/I grounding

### Phase 9: Living Causal Model Era (v14.x)

> Key principle: Compression law binding — bits added < bits saved required for any new axis.

- **v14.0:** Three-Layer Genome + Empirical Coupling + Residual-Gated Axis Admission
- **v14.1:** Dynamic Predictive Genome + Basin-Referenced A + Virtual Trajectory Mode
- **v14.2:** Single Unified Prompt + Migration-Aware Directives

### Phase 10: Structural Inference Era (v15.x)

> Key shift: Relation restructured into two-part adapter. Structural inference at zero token cost before any LLM call.

- **v15.0:** `StructuralRelationEngine` (SRE) + Two-Part Relation Adapter
- **v15.1:** DASS Operator Architecture — SIPA dimensions become proper mathematical operators
- **v15.2:** Real Field Geometry — `Φ` derived from `CompressionOperator` geometry, not designer formula. C_local as trajectory-field alignment signal.
- **v15.3:** Ledger transition begun; `phi_legacy` retained for validation

### Phase 11: Geometric Dynamics Era (v16.x)

> Key shift: The Compressed UII Equation `ẋ = P_M(G⁻¹ · ∇Φ)` is now the explicit dynamical law. Every architectural component maps to one of its terms.

- **v16.0:** Compressed UII Equation + Ledger Architecture
- **v16.1:** `GroundingSpec` + `ground_symbol()` + Bug Fixes
- **v16.2:** Grounding Fallback Clarification
- **v16.3:** `StepLog` Cleanup
- **v16.5:** DASS Convergence Loop (phase-based `step()` with convergence loop replacing fixed 10× batch)

### Phase 12: Continuous DASS Loop Era (v17.x)

> Key shift: 6-phase `step()` architecture eliminated. `S → I → P → A → SMO → S` runs continuously via `iterate()`. CRK becomes purely diagnostic — observes and feeds SMO Repair mode, never gates actions. P's EOG ranking drives commits when it stabilises.

- **v17:** Continuous `iterate()` loop. SMO three modes (Repair/Maintain/Generate). `_commits_to_causality()` requires ranking stability + positive EOG. Migration emergent — no `_should_migrate()` trigger. `run() while True`. `ledger_proximity` channel added. `SymbolGroundingAdapter.ground_symbol()` still used at commit time for fill/type/navigate/evaluate.

### Phase 13: Multi-Agent Field Era (v18.x — landed at v18.9)

**Phase intent (declared at v18.0):** the LLM stops being a tool consulted from outside the loop. It becomes another intelligent substrate inhabiting the same field. LLM outputs enter the Triad's loop as channel readings through `SensingOperator` on equal terms with browser perturbations. The Reality adapter becomes composite. Affordances class into three categories — Environment, Self, Relation. Self-channels feed operator state back into S, taking triadic closure from formal-only to operationally observable. SMO reversibility becomes real (state snapshot + restore on C1/C7) rather than scaffolding-only.

**What v18.0–v18.7 actually were:** the architecture's intent meeting the engineering reality of moving away from a v17 codebase that still organized itself around CRK gating, EOG ranking, SMO repair-then-patch, and an internal math spine. Each sub-version closed off a different piece of that legacy. Each step was correct in isolation; together they were Phase 13 trying to land while still standing on v17 scaffolding. The four structural failures named in v18.1 — parallel-sampling loop, `probe()`-inverts-causal-flow, uncoordinated operator updates, LLM gated outside the loop — were the visible shape of the gap. **v18.8 closed the gap**; v18.9 cleaned up the emission interface; the phase landed.

#### v18.0 — Multi-agent field as environment

- `AgentRealityAdapter` introduced — the only LLM gateway. `CompositeRealityAdapter` wraps Browser + Agent; `iterate()` sees one `RealityAdapter` and S cannot distinguish source.
- New environment channels: `agent_response_latency`, `agent_token_count`, `agent_availability` — LLM perturbations are sensable.
- `AffordanceClassRouter` produces three-class candidate list (Environment / Self / Relation) before `P.rank_actions_by_eog()`.
- Self-channels (`self_s`, `self_i`, `self_p`, `self_a`) feed operator state back into S.
- `agent_registry` passed to CRK for MAC-FCK C6 evaluation; LLMs flagged `is_llm=True` so full MAC-FCK skipped for stateless agents.
- Commit predicate adds `C_global > 0` as first condition. `StateTrace` initializes `c_global = 1.0` (system instantiated as in-protocol per Spine §11).

#### v18.1 — Close-out refinements + four structural failures named for v19

- Protocol exit at `iterate()` level when `C_global ≤ 0` (`raise ProtocolExit`; finally block fires for FAO + ledger save). Stripped in v18.3.
- `api_llm` coverage-based budget degradation — cost as substrate pressure.
- `PeakOptionalityTracker` zero-matrix call removed from commit path.
- `self_p` baseline → running max (geometry-discovered ceiling, replaces fixed 0.7 default).
- SMO Generate stripped (UK-0 minimum: Repair + Maintain only). Generate logic preserved in FAO `AxisAdmissionTest` at session end.
- `requester_frame` + `task_library` seeds in ledger.

**Four structural failures named** (these become the v18.8 work):

1. **FAILURE 1:** Loop is parallel sampling, not reflexive recursion. Trajectory `ẋ` is uncoordinated sum of operator updates; nothing enforces alignment with `∇Φ`.
2. **FAILURE 2:** `probe()` inverts causal flow. Sensing should be substrate-driven, not loop-pulled.
3. **FAILURE 3:** Operator updates are uncoordinated — local bounds preserved, joint loop invariants emergent rather than constructed.
4. **FAILURE 4:** LLM gated outside loop instead of perturbing through S — `SymbolGroundingAdapter` still bypasses S even though `AgentRealityAdapter` exists alongside it.

#### v18.2 — Self / Relation affordance class operational

- Self affordances live: `read_ledger`, `write`, `write_outreach`. Memory expressed as state deformation surfaces.
- Relation affordances live: `ground_triplet`, `revise_triplet`, `promote_basin`, `demote_basin`. Semantic attractors and basin classification become a structure the system can write to and read from.
- `requester_frame` / `task_library` / `output_format` from v18.1 seeds become functional through `_render_query_agent_prompt`.
- `write_outreach` is logged-only (comm-channel mechanism unimplemented).

#### v18.3 — `ProtocolExit` raise stripped

Recognized that a self-validity detector terminating execution based on the implementation's measurement of itself is incoherent with the descriptive math spine. `C_global > 0` holds by construction in a UK-0-correct implementation; doesn't need to be enforced via `raise`.

`raise ProtocolExit` removed from `iterate()`; class, except clause, and "Loop dies if `C_global ≤ 0`" banner left as defensive stubs.

#### v18.4 — `SymbolGroundingAdapter` excised + SMO reversibility actualized

FAILURE 4 partially executed: `SymbolGroundingAdapter`, `ground_symbol()`, and `ground_trajectories()` removed from executable code. `AgentRealityAdapter` is now the only LLM gateway in the running system.

SMO reversibility actualized: U is supposed to be bounded AND reversible per UK-0. Boundedness existed; reversibility was scaffolding-only. v18.4 added a real revert mechanism: `_snapshots: deque(maxlen=10)` holds prior state per iteration. `revert_re_anchor` on C1/C7. `revert_recalc` on C2/C3.

#### v18.5 — Three surgical edits — C5 restored, commit predicate corrected, `ProtocolExit` fully excised

- C5 (External Constraint Attribution) restored to `CRKMonitor.observe()`. Triggers when optionality has dropped AND environment channels show external pressure OR internal consistency is reasonable. Repair directive: `reclassify_external`.
- Load-bearing presence assertion at module load. `_CRK_OBSERVE_CONSTRAINTS_EVALUATED` populated by inspecting `CRKMonitor.observe()` source via `inspect.getsource` at import time. If any of C1–C7 is missing, import fails.
- Commit predicate replaced from `any(EOG > 0)` to argmax stability over `_RANKING_STABLE_NEEDED = 3` consecutive iterations AND best `EOG > 0`.
- `ProtocolExit` class definition, except clause, banner, and `protocol_exit` branch in session-end logging deleted. Corpse fully excised.

#### v18.6 — Working snapshot prior to surgical excision pass

Working state preserved as v18.6 prior to scoped excision. No functional changes from v18.5. Migration code and `CouplingMatrixEstimator` cascade still present but no longer load-bearing.

#### v18.7 — Migration code + coupling estimator cascade excision

Two surgical strips on the v18.6 baseline. **−404 lines total** across the six files.

- `MigrationAttempt` dataclass + `migration_history` field + 4 helper methods + execute branch + measure_channels handling block + FAO migration_history processing block all removed.
- `CouplingMatrixEstimator` class removed entirely; `coupling_estimator` parameter cascade through `PhiField` / `CRKMonitor` / `PredictionOperator` / `ResidualExplainer` / `FAO` removed.
- `c1_stability_risk` SIPA-quadratic block removed from `CRKMonitor.evaluate_pre_action`.
- `_get_predicted_delta` + `action_substrate_map` fallback table + `_predict_delta` + `_last_predicted_delta` + `_try_coupling_refinement` + `REFINEMENT_THRESHOLD` all removed.
- Hardcoded version strings stripped from log payloads.
- `CoherenceOperator` re-anchor pass attempted during v18.7 development and discarded as drift; trustworthy v18.7 baseline = v18.6 + migration strip + coupling strip.

#### v18.8 — Triadic closure becomes structural. The four structural failures from v18.1 are resolved.

**What changes:** the substrate becomes a pure closure loop. The math spine moves out of the substrate entirely. CRK is removed as a separate module. The commit gate moves into Coherence as a structural pattern-match. SMO is rebuilt as a closure-residual emitter. Strict I/O is enforced at every operator boundary. Prediction is stateless. The triadic mapping `T(x) = f_rel(f_self(x), f_env(x))` is no longer a math-spine constraint applied from outside — it is what the closure loop computes structurally on every iteration through compression's edge formation across self / env / affordance / relation channels.

This is what triadic closure was meant to be — across the whole project arc, not just within Phase 13. The result is the base version of an adaptive system in this architecture.

##### Math spine externalized to `uii_observer.py`

- New file `uii_observer.py` holds Φ, ∇Φ, Hessian, Σ_P, vol_opt, C_local, C_global. Not imported by `uii_geometry`, `uii_operators`, `uii_reality`, `uii_fao`, `uii_ledger`. Used by `uii_triad.py` for periodic geometry log lines only.
- `PhiField` (`phi`, `gradient`, `compute_hessian`) computes from substrate emission only. `_build_sigma_p` constructs Σ_P directly from compression's `causal_graph` and sensing's coverage — pure function of emitted state.
- `StateTrace` in `uii_observer` holds the full `compute_c_local` (cosine alignment between `∇Φ` and channel `last_delta`) and the running `c_global` mean. `uii_geometry.StateTrace` is a stub holding only `c_global = 1.0`.

##### CRK removed from the substrate

`CRKMonitor` class, `observe()`, the C1–C7 firing logic, `repair_directives`, the `_CRK_OBSERVE_CONSTRAINTS_EVALUATED` load-bearing presence assertion (v18.5), `evaluate_pre_action` — all gone from the codebase. There is no separate auditor reading operator state and emitting repair directives. The substrate runs; the math spine reads it.

##### Commit gate moves into Coherence as structural pattern-match

`coherence.commit_decision` is the only commit gate. It is an affordance name (a string) or `None`.

The audit per affordance: skip if no outgoing edges in compression's graph (affordance not integrated into `f_rel` — outside reachable state space). On direction-bearing channels (trajectory_direction magnitude above 10% of strongest direction with absolute floor `STABLE_CHANNEL_THRESHOLD = 1e-3`), count sign-matches and sign-mismatches between projection entries and trajectory direction. An affordance passes when it has zero mismatches and at least one match.

The commit decision is the alphabetically-first passer when `signature_deviation < DEVIATION_THRESHOLD = 0.5`; `None` otherwise. The alphabetical tiebreaker is non-optimizing — the architecture forbids gradient-following on a scalar inside the substrate, so the tiebreaker is deterministic but arbitrary among passers.

**What's gone:** EOG ranking entirely. `rank_actions_by_eog`, `expected_optionality_gain`, `score_actions`, `test_virtual`, `_commits_to_causality_from_candidates`, `_select_committed_action`, `_RANKING_STABLE_NEEDED`, `_ranking_history`, `_last_ranking`, `_MIN_ITERS_BEFORE_COMMIT` — all removed. There is no scoring inside the substrate.

##### SMO redesigned as closure-residual emitter

SMO becomes the time-derivative operator on state. `apply(coherence)` computes `δf_rel` — the per-channel closure residual at the state level — for sensing-surface channels.

- On commit: `δf_rel = prediction.affordance_projections[committed]`.
- On no-commit: `δf_rel = prediction.next_delta`.
- Per-channel clipped to `[-MAX_DELTA, MAX_DELTA]` with `MAX_DELTA = 1.0`.
- `cumulative_delta` maintains the per-channel running sum since boot.
- `relation_signals` are emitted with `magnitude = cumulative` so sensing's standard cross-iteration `last_delta` computation yields `δf_rel(this iteration)` on each `relation/{cid}` channel.

**Reversibility is compositional, not stored.** Pre-v18.8 SMO held a `_snapshots` deque and ran `revert_re_anchor` / `revert_recalc` when CRK fired. v18.8 SMO holds no snapshots and runs no reverts. On no-commit, the world has not moved; on commit, it has, and next iteration's sensing metabolizes the consequences.

##### Strict I/O at every operator boundary

Each operator's `apply()` takes a single positional argument that is the prior operator's output:

```python
sensing'    = sensing.apply(env_signal)
compression'= compression.apply(sensing_history)
prediction' = prediction.apply(compression')
coherence'  = coherence.apply(prediction')
smo'        = smo.apply(coherence')
```

Carriers propagate forward — compression holds `active_channel_state`, prediction holds compression by reference, coherence holds prediction by reference. Downstream operators do not reach back through the loop. (FAILURE 1 closed.)

##### Substrate-driven sensing

`env_signal` accumulates each iteration from: composite reality probe (browser + agent + api_llm budget channel), `_build_self_channel` (Python introspection), `_build_affordance_channel` (per-affordance availability + just-used spike), `_build_ledger_channel` (proximity to inherited operator_snapshot), and SMO's `relation_signals` stashed from the previous iteration.

Sensing reads `env_signal` once and produces the perceptual surface; the rest of the loop follows. Probing happens once per iteration, before sensing runs — sensing is the integrator, not the puller. (FAILURE 2 closed.)

Affordances are sensing channels. Each affordance becomes a channel reporting `coverage = 1.0` if available, `magnitude = rate = 1.0` if just-committed last iteration, else 0. Compression's normal co-movement detection learns affordance→outcome edges naturally — actions become first-class causal sources because they are sensing channels, not because of any special-case attribution path.

##### Self channels as Python introspection

`_build_self_channel` emits Python introspection signals each iteration:

- `self/clock/wall_time`, `self/clock/iteration`, `self/clock/iter_latency`
- `self/process/cpu_percent`, `self/process/memory_rss`, `self/process/num_fds` (psutil; gracefully degrades if unavailable)
- `self/gc/gen0_count`, `self/gc/gen1_count`, `self/gc/gen2_count` (Python GC generations)

`self/...` hierarchical naming marks the class at id level. Channels are instantiated dynamically by sensing's signal-arrival mechanism — no pre-allocated schema; if a new introspection signal becomes available it simply emits and the channel comes into being. Compression discovers cross-class edges between self channels and env / affordance channels naturally.

v18 `self_s`, `self_i`, `self_p`, `self_a` (scalar projections of operator state fed back into S) is superseded. The substrate's introspection is the substrate's own runtime conditions — clock, memory, GC pressure — not scalar reductions of its own operators.

##### Compression's surprise-derived alpha

Edge plasticity is now derived inside compression rather than set by a global SMO knob (v15.1's `smo_epsilon = 0.1` is no longer the controlling alpha).

```
active_count = len(active_state)
active_with_motion = sum(1 for ch in active_state.values()
                          if abs(ch.last_delta) > _noise_floor(ch.id))
surprise_ratio = active_with_motion / active_count
alpha = clip(0.05 + 0.45 * surprise_ratio, 0.05, 0.5)
```

Three-phase compression each iteration: predict per-target deltas + record residuals (Phase 1), update edge weights/confidences from sign agreement on observed deltas (Phase 2), propose-and-prune candidate new edges from co-movement (Phase 3).

Residuals are compression's internal machinery for substrate-derived alpha selection — they are not `δf_rel`. `δf_rel` is computed in SMO at the state level over prediction's projections.

##### `PredictionOperator` stateless

Prediction is a pure function of compression at this iteration. It maintains no history and no accuracy estimates. `apply(compression)` produces:

- `next_delta`: forward pass through `causal_graph` with current channel `last_delta` values and no commit hypothesised — the trajectory's natural next motion.
- `affordance_projections[a]`: forward pass with `hypothetical_deltas[a] = UNIT_COMMIT_MAGNITUDE = 1.0` — the per-target delta if affordance `a` commits this iteration.

Self-grading would put a CRK-shaped quantity inside the substrate; the math spine's accuracy is computed externally against the substrate's emitted state transitions.

**What's gone:** `rank_actions_by_eog`, `test_virtual`, `observe_outcome`, `covariance_matrix`, `simulate_covariance_update`, `configuration_vector`. `to_grounded_proxy` is preserved as the projection used by `SubstrateState.P` and external observation tooling.

##### Trust signal redefined

`_update_agent_registry` uses `coherence.consistency.loop_closure` at the moment of commit as the trust signal. `loop_closure` is the geometric mean of the four pair consistencies (s↔i, i↔p, p↔a, smo).

High `loop_closure` at commit means the substrate's perception, integration, projection, and audit aligned when this agent was queried; low means they didn't. This is observational, not optimised — the substrate does not select agents to maximize trust.

##### Summary: the four structural failures named in v18.1

- **FAILURE 1 (parallel sampling):** closed by strict I/O at every operator boundary. The loop is reflexive recursion.
- **FAILURE 2 (probe inverts causal flow):** closed by `env_signal` accumulation. Probing happens once per iteration before sensing runs.
- **FAILURE 3 (uncoordinated operator updates):** closed by carriers. Joint loop invariants are constructed in the carrier graph.
- **FAILURE 4 (LLM gated outside loop):** closed by `AgentRealityAdapter` being the only LLM gateway and its outputs entering as channel readings through composite probe.

#### v18.9 — Phase 13 close-out: the substrate's only output is its causality

**Architectural commitment:** the substrate emits one JSONL record per iteration on stdout, and that record is `{cause, ẋ}` — what the substrate pushed into the world (`commit`) and what moved as a result (`delta_f_rel`, the per-channel bounded motion this iteration). Math-spine readings, ceremonial events, periodic summaries, status prints — all moved out of the substrate. The loop emits its causal signature; readers external to the loop interpret. No content other than causality is permitted by the architecture's invariants.

##### Causal emission: the substrate's only output

Every `iterate()` call ends with:

```python
sys.stdout.write(json.dumps({
    'iter':        self._iteration_count,
    't':           time.time(),
    'commit':      committed_action,   # action_dict-as-dispatched, or None
    'delta_f_rel': delta_f_rel,        # {channel_id: delta} non-zero only
}, default=_json_default) + '\n')
sys.stdout.flush()
```

- `commit` is the `action_dict`-as-emitted (the cause the substrate pushed into the world) or `null` when no commit fired.
- `delta_f_rel` is the closure residual SMO produced this iteration, per channel, non-zero entries only — the substrate's perception of its own motion.
- Computed against substrate emission, not by re-running SMO's internal logic; `new_cumulative − prior_cumulative` is exactly the bounded delta SMO clipped this iteration.

**stderr carries execution faults** (Python tracebacks, adapter errors). The JSONL stream on stdout is never interrupted.

##### `uii_fao.py` stripped entirely

`FailureAssimilationOperator`, `ResidualTracker`, `ResidualExplainer`, `AxisAdmissionTest`, `ProvisionalAxisManager` — all removed. Parallel-curator scoring of the substrate's residuals on its own behalf is an architectural inversion: a separate auditor watching the substrate's perception of motion and deciding which channels deserve to become axes is the same shape as the CRK auditor watching the substrate's coherence and deciding which constraints fire. Both are external scoring that re-enters the substrate's update logic; both are out.

What FAO did at session end (axis admission via four-pass test) is now external work on the JSONL stream. v19's interface is where structural readings of the trajectory live.

##### Other v18.9 strips

- `_log_geometry()` removed from `uii_triad`. `uii_observer` is no longer imported by the substrate; the math-spine module exists for external tooling only.
- `residual_tracker` block in `iterate()` removed (FAO is gone; nowhere to feed).
- Session_end / session_start ceremonial events removed.
- Per-commit and per-period status prints removed.
- Log payloads no longer tag version strings.

##### What persists across sessions

Ledger persistence stays in the `__main__` finally block (graceful Ctrl+C). A crash still loses post-last-save deltas — that's a v19+ persistence design question.

##### What Phase 13 close means

Per source-of-truth §11, instantiated systems are in-protocol by construction. The loop either runs or it doesn't; whether the running loop is in regime is what an external observer reads from the emitted state-trajectory. v18.9 makes the emission clean enough that the external observer is a real possibility, not an aspiration. With the emission port stable, Phase 14 can begin.

### Phase 14: Interface Era (v19.x — beginning)

> Working name. Phase 13 dissolved the LLM/substrate boundary; Phase 14 establishes explicit interface surfaces around a substrate that previously had none.

**Phase intent.** The substrate runs, emits, and metabolizes. With v18.9 it has a clean causal emission — `{commit, delta_f_rel} = {input, ẋ}` per iteration, public, structural, forced by the architecture. v19 adds the matching surfaces: external entities (humans, peer systems, monitoring tools) can perturb the substrate discontinuously through forward transducers, and external readers can interpret the trajectory through pure JSONL consumers. The math spine's ẋ becomes a public artifact (v18.9), and Phase 14 builds the interfaces that let other systems push cause in and read trajectory out.

Two structural commitments distinguish Phase 14:

1. **Discontinuous perturbation through forward transducers.** Until v19 the triad probed reality on its own schedule — a browser fetch, an LLM call. As of v19 the architecture admits external input that perturbs the triad on the *perturber's* schedule. The forward transducer adapts external input (a typed line, a queued message, a sensor event) into the existing `env_signal` accumulation pathway. The substrate's strict-I/O contract is unchanged — input arrives the same way every other channel arrives — but the loop is no longer the sole source of timing for what enters sensing.
2. **External observation through pure JSONL consumers.** A reader external to the loop, importing nothing from the substrate, can ingest the JSONL stream and report descriptive features of trajectory shape — bounded? metabolized? continuous? — without ever opening the operator state. The doctor reads the patient's trajectory; the doctor never opens the patient up.

#### v19.0 — Bones

**What's added.** The observability utility (3 files), separate from the substrate, never imported by it. Architectural commitment to forward transducers stated; specific transducer implementations are deployment-specific and not in the v19.0 substrate.

##### `observability.py` — pure JSONL consumer

- Reads `{iter, t, commit, delta_f_rel}` and applies math-spine *shape* predictions to the trajectory.
- Imports nothing from `uii_*`. Stdlib only.
- Does NOT compute Φ, ∇Φ, vol_opt — those need operator-internal state. What it does instead is test trajectory shape against three qualitative predictions the math implies for closure-holding systems under perturbation:

  - **Bounded.** `ẋ ≈ η(t)` when closure holds. A bounded perturbation should produce bounded `δf_rel` through the iteration it lands and shortly after. A spike beyond an envelope of pre-perturbation baseline indicates closure stress — the gradient term has woken up because the residual grew beyond metabolic.
  - **Metabolize.** A discrete perturbation gets integrated by compression across iterations. Its `δf_rel` signature should decay with a characteristic profile — strongest at N+0..N+1, fading as the structure absorbs into `f_rel`. No decay means runaway; instant flat-zero means the perturbation never landed.
  - **Continuity.** Successive states preserve core structure. Catastrophic `δf_rel` jumps between consecutive iterations indicate trajectory discontinuity.

- **Shape-mismatch is symmetric.** A divergence between predicted and observed shape could mean closure is breaking *or* the user's prediction was bad. The library reports descriptive features and lets the user read the gestalt; it does not take sides.
- **Predictions are descriptive, not verdictal.** No "the triad was coherent" or "the triad failed." The user reads the gap.

Default qualitative-shape thresholds:

```
DEFAULT_BASELINE_WINDOW    = 30    # iterations of pre-perturbation history
DEFAULT_ENVELOPE_SIGMAS    = 3.0   # bounded-ness threshold (× baseline σ)
DEFAULT_METABOLIZE_HORIZON = 5     # iterations to expect decay-to-baseline
DEFAULT_RETURN_SIGMAS      = 1.0   # "returned to baseline" threshold (× σ)
DEFAULT_CONTINUITY_SIGMAS  = 5.0   # catastrophic-jump threshold (× σ)
```

These pick concrete values for what the math says qualitatively (bounded, decays, continuous). Calibration against an actual session may tighten them.

##### `cli_observe.py` — terminal-side observer

Tails a substrate JSONL stream (file or stdin) and renders per-iteration trajectory + math-spine shape analysis on completed perturbation events. Optional sidecar perturbation file (`TIMESTAMP\tLABEL` per line) for cross-process correlation when a forward transducer in another process appends a line each time it sends user input.

##### `README.md` — architectural commitments + integration sketch

States the JSONL-only access constraint, the shape-prediction methodology, the doctor/patient frame, and a sketch of single-process and cross-process integration paths. Does not include a finished forward-transducer implementation — that's deployment-specific.

##### What v19.0 does NOT yet have

- A finished `UserRealityAdapter` or equivalent forward-transducer adapter. The substrate still has Browser + Agent in its `CompositeRealityAdapter`; user input as a structured channel arriving through `env_signal` is the architectural target, not yet the running code.
- A persistence story richer than ledger-on-Ctrl+C. v18.9's note carries: a crash still loses post-last-save deltas.
- Schema reconciliation between substrate emission (`delta_f_rel` as `Dict[str, float]`) and observability ingestion (`delta_f_rel` as scalar `float`). The library's dataclass parse is `float(d['delta_f_rel'])` — works for a reduced/normed version, doesn't work for the raw dict the substrate emits. v19+ design question.

##### Why v19.0 is "bones"

The library and CLI are functional. The integration example is a sketch — substrate construction is deployment-specific and not filled in. The success criterion (perturbation → metabolization → shape match) is exercisable in any environment that can construct a running substrate. The remaining v19.x work is the forward transducer, the schema reconciliation, the persistence story, and calibration of the shape-prediction defaults against real sessions.

---

## iterate() Architecture (v18.9/v19.0 — strict I/O closure loop with JSONL emission)

`iterate()` runs every cycle unconditionally. The body is the strict I/O wiring: each operator's `apply` takes a single positional argument that is the prior operator's output. The iteration ends with one JSONL record on stdout.

```python
_t0 = time.time()
_iteration_count += 1
temporal_memory.decay_all()

# 1. probe + accumulate env_signal
env_signal = composite_reality.probe()
current_affordances = composite_reality.get_current_affordances()
env_signal.update(_build_ledger_channel())     # proximity to inherited peak
env_signal.update(_build_self_channel())       # Python introspection
env_signal.update(_build_affordance_channel(current_affordances))
if _pending_relation_signals:                  # SMO from previous iter
    env_signal.update(_pending_relation_signals)
    _pending_relation_signals = {}

# 2. S → I → P → A → SMO  (strict I/O)
new_sensing     = state.sensing.apply(env_signal)
_sensing_history.append(new_sensing)
new_compression = state.compression.apply(_sensing_history)
new_prediction  = state.prediction.apply(new_compression)
new_coherence   = state.coherence.apply(new_prediction)

# Capture prior cumulative_delta before SMO replaces it; the
# difference (new − prior) IS this iteration's δf_rel — emitted on stdout.
prior_cumulative = dict(smo_v151.cumulative_delta)
new_smo          = smo_v151.apply(new_coherence)

_pending_relation_signals = new_smo.relation_signals
state    = SubstrateState(new_sensing, new_compression,
                           new_prediction, new_coherence)
smo_v151 = new_smo

_iteration_times.append(time.time() - _t0)

# δf_rel(this iteration) — non-zero entries only, computed against emission.
delta_f_rel = {}
all_channels = set(new_smo.cumulative_delta.keys()) | set(prior_cumulative.keys())
for cid in all_channels:
    d = (new_smo.cumulative_delta.get(cid, 0.0) - prior_cumulative.get(cid, 0.0))
    if d != 0.0:
        delta_f_rel[cid] = float(d)

# 3. commit gate — coherence.commit_decision is the answer
committed_action = None
commit = new_coherence.commit_decision
if commit is not None:
    viable_now = set(current_affordances.get('viable_action_types', set()))
    if commit in viable_now:
        grounding_spec = _build_grounding_spec()
        action_dict    = _action_dict_from_type(commit, current_affordances,
                                                 check_temporal=True,
                                                 grounding_spec=grounding_spec)
        _pending_action_attribution = action_dict.get('type', 'observe')
        pre_ch, post_ch, ctx = composite_reality.execute(action_dict)
        if commit == 'query_agent':
            _update_agent_registry(action_dict, ctx)
        _commit_count   += 1
        committed_action = action_dict

# 4. structural emission — one JSONL record on stdout
sys.stdout.write(json.dumps({
    'iter':        _iteration_count,
    't':           time.time(),
    'commit':      committed_action,
    'delta_f_rel': delta_f_rel,
}, default=_json_default) + '\n')
sys.stdout.flush()
```

### Notes on key elements

- **`env_signal` is the only path into sensing.** Composite reality probe + self introspection + affordance channels + ledger proximity + SMO's previous-iteration `relation_signals` all enter as flat `{cid: {magnitude, rate, coverage}}` entries. Sensing reads them all the same way; compression discovers cross-class edges between them. *Phase 14 commitment: forward transducers (deployment-specific) extend this set with external-input channels under the same I/O contract.*
- **`_pending_relation_signals`** is the only piece of state that survives between iterations on the SMO side. SMO's `relation_signals` from iteration N enter sensing on iteration N+1 through `env_signal`, and sensing's standard `last_delta = magnitude − prior_magnitude` yields `δf_rel(this iteration)` on each `relation/{cid}` channel.
- **`_pending_action_attribution`** — the action just committed; consumed by the next iteration's `_build_affordance_channel` to mark the affordance as just-used via a one-iteration magnitude/rate spike. This is how committed actions enter compression's edge discovery as first-class causal sources.
- **Commit dispatch.** Coherence's `commit_decision` is consulted directly. No external scoring. The `viable_now` check is a defensive guard against affordance availability changing between when sensing built the affordance channel and when execute fires.
- **The JSONL emission is the loop's only output, and it is the loop's causality.** No conditional log gates, no math-spine readings, no banners. One record per iteration, always. `commit` is the cause; `delta_f_rel` is ẋ. Readers external to the loop interpret.

### `run()` (v18.9 — no termination condition)

```python
while True: self.iterate()
except KeyboardInterrupt: ...
finally: ledger.save_ledger()    # FAO.distill_to_ledger() removed in v18.9
```

---

## Migration Trigger (still fully stripped)

As of v19.0, no `migrate` affordance exists in the running system. The `MigrationAttempt` dataclass, the `migrate` action_type branch, the `migrate_outcome` handling block, the `migration_history` field, and the FAO `migration_history` processing block remain absent (stripped in v18.7; FAO itself stripped in v18.9).

If `migrate` is reintroduced, it must be defined through the current channel-based affordance routing (`BASE_AFFORDANCES` / `SELF_AFFORDANCES` / `RELATION_AFFORDANCES`), not via the v17 dataclass + classifier + delta-table pipeline. v18.8's commit pathway means a reintroduced `migrate` would need to be projected by `PredictionOperator`, audited by `CoherenceOperator`, and committed structurally — like every other affordance.

No `boundary_pressure`. No `DeathClock`. No `vol_opt` decline window check. Resource pressure sensed via `api_llm` channel coverage in `CompositeRealityAdapter.probe` (built from `agent._cumulative_tokens` against `UII_API_LLM_BUDGET`).

---

## Key Components (v19.0)

| Component | Role / Status |
|---|---|
| `SubstrateState` | Per-iteration immutable container holding S, I, P, A operator instances. Replaced atomically each `iterate()` call. S/I/P/A scalar properties expose `to_scalar_proxy` / `to_grounded_proxy` projections for external observation. The substrate itself does not consume these scalars. |
| `StateTrace` (uii_geometry) | v18.8+: substrate-side stub holding only `c_global = 1.0`. Reality adapters and a few logging paths read this; the substrate does not read gradient-vs-trajectory alignment. |
| `StateTrace` (uii_observer) | Full version with `compute_c_local` (cosine alignment between `∇Φ` and channel `last_delta`) and running `c_global` mean. Computed externally against substrate emission. Not consumed by the loop. |
| `SensingOperator` | `apply(env_signal) → SensingOperator'`. Reads flat dict `{cid: {magnitude, rate, coverage}}`. `last_delta` computed cross-iteration as `magnitude − prior_magnitude`. Coverage decay 0.9 toward 0 on absent signal; channels deactivate after `INACTIVITY_THRESHOLD = 50` iterations. New channels instantiated dynamically by signal arrival. Channel registry monotonic — channels never removed. |
| `CompressionOperator` | `apply(sensing_history) → CompressionOperator'`. Three-phase per-iteration: predict per-target deltas + record residual variance; edge weight/confidence updates from sign agreement; propose-and-prune candidate edges from co-movement. `alpha` derived from `surprise_ratio`. `HISTORY_WINDOW=10`. `MAX_ACTIVE_EDGES=200`. `active_channel_state` snapshot held as carrier for prediction. |
| `PredictionOperator` | `apply(compression) → PredictionOperator'`. Stateless across iterations. `next_delta`: forward pass through `causal_graph` with current channel deltas, no commit hypothesised. `affordance_projections[a]`: forward pass with `hypothetical_deltas[a] = UNIT_COMMIT_MAGNITUDE = 1.0`. v18.8: `rank_actions_by_eog` / `test_virtual` / `observe_outcome` / `covariance_matrix` all removed. |
| `CoherenceOperator` | `apply(prediction) → CoherenceOperator'` carrying `commit_decision`. `trajectory_direction`: per-channel EMA of `last_delta` (`EMA_ALPHA=0.05`). `loop_signature`: EMA of `{active_channels, graph_edges, mean_confidence}`. `signature_deviation`: deviation of current iter from running signature. Audit per affordance: skip if no outgoing edges; on direction-bearing channels count sign-matches/mismatches. Pass = 0 mismatches AND ≥1 match. `commit_decision = alphabetically-first passer` when `sig_dev < 0.5`, else `None`. **IS the commit gate.** No external scoring. |
| `SelfModifyingOperator` | `apply(coherence) → SelfModifyingOperator'`. Closure-residual emitter. On commit: `δf_rel = prediction.affordance_projections[committed]`. On no-commit: `δf_rel = prediction.next_delta`. Per-channel clipped to `[-MAX_DELTA, MAX_DELTA]` with `MAX_DELTA = 1.0`. `cumulative_delta`: per-channel running sum since boot. `relation_signals`: emitted with `magnitude=cumulative`; re-enter sensing through `env_signal` next iter. v18.8: no Repair/Maintain/Generate modes; no `_snapshots`; no revert methods; no `recent_external_attribution`; no `predicted_delta` input. |
| `PhiField` (uii_observer) | `Φ = α·C + β·log(O) + γ·K`. `compute_hessian` decomposes `H = α·H_C + β·H_O + γ·H_K + ε·I`. Σ_P built directly from compression's `causal_graph` and sensing's coverage by `_build_sigma_p`. K-component anchored to `ledger.operator_snapshot` peak coverage. v18.9: external to substrate; not imported by `uii_triad`. |
| `eigen_decompose` (uii_observer) | Symmetric eigendecomposition with regularization fallback. Module-level utility. |
| `GroundingSpec` | Built at action commit by `_build_grounding_spec`. Carries `desired_delta`, `dark_channels`, `top_gradient_channels`, `current_url`, `page_title`, `nat_grad_magnitude`. v18.8+: built from substrate-internal channel mass — math spine quantities not consulted; the substrate shapes its own grounding from its own perception. |
| `PeakOptionalityTracker` | Records `hessian_snapshot` + `operator_snapshot` at peak vol_opt. v18.9: called from external observation tooling, not from inside the substrate. |
| `TriadLedger` | Eleven fields: `hessian_snapshot`, `operator_snapshot`, `causal_model`, `discovered_structure`, `geometry_history`, `semantic_attractors`, `basin_classification`, `agent_registry`, `requester_frame`, `task_library`, `output_format`. v18.8: `agent_registry` records `loop_closure` history. v18.9: `discovered_structure` is no longer written by FAO (FAO is gone); the field remains for inheritance from prior runs. |
| `AgentRealityAdapter` | Sole LLM gateway. `execute()` → `llm_client.call()` produces `(response, tokens)`. Probe builds `agent_response_latency` / `agent_token_count` / `agent_availability` via event-buffer aggregation. `_response_history` holds prior responses. `_cumulative_tokens` drives `api_llm` coverage in composite probe. |
| `CompositeRealityAdapter` | Wraps Browser + Agent. `probe()` merges both + adds `api_llm` channel. `execute()` routes by action_type: `query_agent` → agent; `ground_*` / `revise_*` / `promote_*` / `demote_*` → `_execute_relation`; `read_ledger` / `write` / `write_outreach` → `_execute_self`; default → browser. *Phase 14 architectural target: extend with forward-transducer adapter for external input.* |
| `BrowserRealityAdapter` | Playwright DOM perturbations. `get_current_affordances` returns links/buttons/inputs/readable/url/title/scroll. Migrate branch + helpers stripped (v18.7). |
| `AttractorMonitor` | Classifies `semantic_attractors` basin type (Objective/Subjective) from edge confidence + `hessian_snapshot.vol_opt` + `signature_deviation`. Used by `_execute_relation` when `ground_triplet` commits. |
| `MentatTriad` | Owns per-iteration `SubstrateState` and persistent SMO instance. `iterate()` runs strict-I/O closure loop, ends with one JSONL record on stdout. `run()` loops until `KeyboardInterrupt`. v18.9: no `_log_geometry`, no ceremonial events, no per-commit prints. `smo_v151` attribute name retained for backward compatibility. |
| `TemporalPerturbationMemory` | Bounded buffer (window=5, capacity=20) for trajectory analysis. `decay_all()` called every iteration. |
| **`observability.JSONLRecord`** *(v19.0)* | Dataclass for one substrate emission record: `{iter, t, commit, delta_f_rel}`. `from_line()` parses JSONL; returns `None` on parse failure. Schema = v18.9. |
| **`observability.Baseline`** *(v19.0)* | Rolling window of `δf_rel` values. Provides mean, std, envelope. Default 3σ. |
| **`observability.PerturbationEvent`** *(v19.0)* | Tracks one perturbation across its observation horizon. Records pre-baseline + observed trajectory; renders descriptive shape analysis (bounded? decayed? continuous?). |
| **`observability.ObservabilityEngine`** *(v19.0)* | Pure-stream consumer. Ingests JSONL records, maintains baseline, marks perturbations from sidecar or in-process calls, completes events when their horizon elapses. No substrate access. |

---

## Eliminated in v18.x–v19.0 — Do Not Re-introduce

### v18.1
- `raise ProtocolExit` at `iterate()` level (added v18.1, stripped v18.3)
- SMO Generate mode in main loop (full `AxisAdmissionTest` moved to FAO at session end; FAO removed v18.9)
- `self_p` baseline as fixed `0.7` default (replaced by running max from observed P values)
- `PeakOptionalityTracker` zero-matrix call inside commit path (moved to `_log_geometry()`; `_log_geometry()` itself removed v18.9)

### v18.3
- `raise ProtocolExit` on `c_global ≤ 0` — self-validity detector terminating execution based on implementation's measurement of itself was incoherent with descriptive math spine.

### v18.4
- `SymbolGroundingAdapter` class
- `ground_symbol()`, `ground_trajectories()` methods
- `llm_client.call()` outside `AgentRealityAdapter.execute()`
- SMO repair without snapshot reversal — patching corrupted state instead of rolling back

### v18.5
- `ProtocolExit` class definition (corpse from v18.3 strip)
- `except ProtocolExit` branch in `run()`
- "Loop dies if `C_global ≤ 0`" banner line
- `exit_reason = {'kind': 'protocol_exit', ...}` session-end JSON branch
- `any(EOG > 0)` commit predicate — fired commits before ranking converged

### v18.7
- `MigrationAttempt` dataclass; `self.migration_history` field
- `_run_migration_code`, `_classify_migration_outcome`, `_migration_delta_from_outcome`, `_snapshot_substrate` methods
- `migrate` action_type branch in `BrowserRealityAdapter.execute`
- `migrate_outcome` handling block in `measure_channels`
- FAO `migration_history` parameter + 47-line processing block in `distill_to_ledger`
- `hashlib` import (`uii_reality.py`)
- Hardcoded version strings in log payloads (4×)
- `CouplingMatrixEstimator` class — including `update`/`observe`/`merge`/`to_ledger_entry`/`from_ledger_entry`
- `coupling_estimator` parameter cascade across `PhiField` / `CRKMonitor` / `PredictionOperator` / `ResidualExplainer` / `FAO`
- `coupling_confidence` parameter from `CompositeRealityAdapter.execute` and `BrowserRealityAdapter.execute`
- `c1_stability_risk` SIPA-quadratic block in `CRKMonitor.evaluate_pre_action`
- `_get_predicted_delta` + `action_substrate_map` fallback table; `_predict_delta` + `_last_predicted_delta`
- `_try_coupling_refinement` + `REFINEMENT_THRESHOLD`
- `coupling_matrix` and `action_substrate_map` ledger writes from `FAO.distill_to_ledger`

### v18.8

v18.8 is the largest single-version excision in the project's history. Everything below was removed:

- `PhiField` from `uii_geometry.py` — moved to `uii_observer.py`. Substrate no longer imports it.
- Full `StateTrace` from `uii_geometry.py` — `compute_c_local`, `c_local_history`, `c_global` running mean. Moved to `uii_observer.py`. `uii_geometry.StateTrace` is now a stub.
- `eigen_decompose` from `uii_geometry.py` — moved to `uii_observer.py`.
- `expected_optionality_gain` function — gone entirely. EOG was the substrate's scoring quantity; the substrate no longer scores.
- `PhiField.score_actions` — gone.
- `CRKMonitor` class entirely — `observe()`, `evaluate_pre_action`, `_c5_post`, `_CRK_OBSERVE_CONSTRAINTS_EVALUATED` load-bearing presence assertion, the C1–C7 firing logic, `repair_directives`. The constraint-checker as separate module is gone.
- `PredictionOperator.rank_actions_by_eog` — gone. The EOG-ranking commit pathway is gone with it.
- `PredictionOperator.test_virtual` — gone.
- `PredictionOperator.observe_outcome` — gone. Prediction is stateless across iterations.
- `PredictionOperator.covariance_matrix` / `simulate_covariance_update` / `configuration_vector` — gone.
- SMO Repair / Maintain / Generate modes — gone. SMO is now a single-mode closure-residual emitter.
- SMO `_snapshots` deque — gone (added v18.4). Reversibility is compositional.
- SMO `revert_re_anchor` / `revert_recalc` methods — gone.
- SMO `recent_external_attribution` flag — gone (added v18.5 with C5).
- SMO `predicted_delta` input parameter — gone.
- `AffordanceClassRouter` class + `get_viable_candidates` — gone. Affordances enter as channels through `env_signal`; coherence audits projections.
- `_commits_to_causality_from_candidates` / `_select_committed_action` / `_RANKING_STABLE_NEEDED` / `_ranking_history` / `_last_ranking` / `_MIN_ITERS_BEFORE_COMMIT` — all gone. Coherence's `commit_decision` is the only gate.
- Six-phase `iterate()` body decomposition — replaced by strict-I/O wiring.
- `self_s` / `self_i` / `self_p` / `self_a` as scalar feedback channels — gone (added v18.0). Self channels are now Python introspection signals.
- Three-class candidate list workflow — gone with EOG.
- `agent_registry` `eog_history` field — replaced by `trust_history` from `coherence.loop_closure`.

### v18.9

Phase 13 close-out — emission interface stabilization. Everything below was removed:

- **`uii_fao.py` entirely.** `FailureAssimilationOperator`, `ResidualTracker`, `ResidualExplainer`, `AxisAdmissionTest`, `ProvisionalAxisManager` — all stripped. Parallel-curator scoring of the substrate's residuals is an architectural inversion of the same shape as CRK.
- `_log_geometry()` from `uii_triad`. The math-spine readings logging path is gone; `uii_observer` is no longer imported by the substrate.
- `residual_tracker` block in `iterate()` (FAO is gone).
- `_LOG_GEOMETRY_EVERY` constant.
- Session_start / session_end ceremonial JSON events.
- Per-commit and per-period status prints (commit count, vol_opt summaries, geometry banners).
- Hardcoded version strings in log payloads.
- The `phi_history` parameter on `FAO.distill_to_ledger` (FAO removed; parameter and call site gone).

What was **kept**: ledger persistence in `__main__` finally block (graceful Ctrl+C). A crash still loses post-last-save deltas — v19+ persistence design question.

### v19.0

No new strips. v19.0 is additive — three new files (`observability.py`, `cli_observe.py`, `README.md`) external to the substrate. The substrate's six files inherit from v18.9 with no changes mandated by v19.0 itself.

What v19.0 *does not* add (intentionally):
- A finished forward-transducer adapter. The architectural commitment is stated; specific implementations are deployment-specific.
- Any operator-state query path. Observability stays JSONL-only.
- Any verdict-rendering output ("coherent" / "failed"). Shape predictions are descriptive.

---

## Affordance Sets (v19.0)

Affordances class into three sets, defined in `uii_geometry.py`. Coherence's commit pathway operates over `BASE_AFFORDANCES` — the projection set used by `PredictionOperator`. Self and Relation classes are reachable via `CompositeRealityAdapter.execute` action_type routing but are not currently visible to the commit gate (see [Open Issues](#trajectory-open-issues-carry-into-v191) — scoping gap).

### `BASE_AFFORDANCES` (12 elements; the projection set Prediction operates over)

```python
{'navigate', 'click', 'fill', 'type', 'read',
 'scroll', 'observe', 'delay', 'evaluate',
 'query_agent', 'python', 'llm_query'}
```

> Note — `llm_query` is preserved in `BASE_AFFORDANCES` as a name but the running LLM gateway is `query_agent` (`AgentRealityAdapter.execute`). `llm_query` has no executable handler in any current adapter; it remains in the projection set as a structural placeholder pending consolidation.

### `SELF_AFFORDANCES` (memory expressed as state-deformation surfaces)

```python
{'read_ledger', 'write', 'write_outreach'}
```

Routed through `CompositeRealityAdapter._execute_self`. `write_outreach` remains logged-only (no comm-channel mechanism implemented). Not currently in `PredictionOperator`'s projection set.

### `RELATION_AFFORDANCES` (semantic attractors and basin classification)

```python
{'ground_triplet', 'revise_triplet',
 'promote_basin', 'demote_basin'}
```

Routed through `CompositeRealityAdapter._execute_relation`. `ground_triplet` and `revise_triplet` write through to `ledger.semantic_attractors` with `AttractorMonitor` classification. Not currently in `PredictionOperator`'s projection set.

### `ENVIRONMENT_AFFORDANCES` (broader reality-side menu, used as the executable set for action_type dispatch)

```python
{'navigate', 'click', 'fill', 'type', 'read', 'scroll',
 'observe', 'delay', 'evaluate', 'python',
 'write_file', 'send_email', 'query_agent'}
```

Note: `write_file`, `send_email` are present here but absent from `BASE_AFFORDANCES` — currently sense-able through the reality adapter but not projected by Prediction.

### Channel-channel signal classification (used by AxisAdmissionTest until v18.9)

```python
INTERFACE_COUPLED_SIGNALS = {
  'dom_depth', 'element_count', 'link_count', 'button_count',
  'input_count', 'scroll_position', 'viewport_height',
  'dom_complexity',
}
POTENTIALLY_INVARIANT_SIGNALS = {
  'response_latency', 'content_entropy',
  'surface_change_rate', 'interaction_density',
}
```

These sets are still defined in `uii_geometry` but no longer consumed by the substrate (FAO/AxisAdmissionTest stripped in v18.9). They are preserved for external tooling that may re-run admission tests on the JSONL stream.

---

## File Inventory (v19.0 — nine files)

v19.0 adds three external utility files (`observability.py`, `cli_observe.py`, `README.md`) on top of the v18.9 substrate. The six substrate-and-observer files inherit from v18.9 with no changes mandated by v19.0.

| File | Lines | Role |
|---|---:|---|
| `uii_geometry.py` | 266 | Substrate-side data containers and ABCs only. `SubstrateState`, `StateTrace` stub, `GroundingSpec`, affordance sets, signal classifications, agent/reality ABCs, `QUERY_AGENT_GROUNDING_PROMPT`. |
| `uii_operators.py` | 977 | All five operators under strict I/O. Sensing, Compression, Prediction, Coherence, SMO. `OperatorConsistencyCheck` dataclass. |
| `uii_reality.py` | 956 | `AttractorMonitor`, `AgentRealityAdapter`, `CompositeRealityAdapter`, `BrowserRealityAdapter`, module-level `_build_channel_probes`. |
| `uii_ledger.py` | 426 | `TriadLedger` (eleven-field dataclass), `PeakOptionalityTracker.update()` (takes external H/eigvals/eigvecs), `load_ledger` / `save_ledger`. |
| `uii_observer.py` | 352 | **External math-spine reader.** Not imported by any substrate file in v19.0 (v18.8 had `uii_triad` import it for `_log_geometry`; that was stripped in v18.9). `PhiField`, `StateTrace` (full), `_build_sigma_p`, `eigen_decompose`. |
| `uii_triad.py` | 1062 | `MentatTriad` orchestrator. `iterate()` runs the strict-I/O closure loop and ends with one JSONL record on stdout. `_build_self_channel`, `_build_affordance_channel`, `_build_grounding_spec`. v18.9: no `_log_geometry`, no FAO, no ceremonial events. *(Down from 1301 lines in v18.8.)* |
| **`observability.py`** *(v19.0)* | 462 | **Pure JSONL consumer.** Imports nothing from `uii_*`. `JSONLRecord`, `Baseline`, `PerturbationEvent`, `ObservabilityEngine`. Tunable shape thresholds at module top. |
| **`cli_observe.py`** *(v19.0)* | 220 | CLI for tailing JSONL (file or stdin) with optional sidecar perturbation file. Renders trajectory + completed-event shape analysis. |
| **`README.md`** *(v19.0)* | 210 | Architectural commitments, JSONL access constraints, integration sketches. |

**Removed from inventory in v18.9:**
- `uii_fao.py` (was 527 lines in v18.8) — FAO's residual machinery and axis admission stripped entirely.

---

## Ledger Schema (v19.0)

`TriadLedger` persists across sessions. Eleven fields. Inherited at startup; updated during run; written to disk in `finally` block.

| Field | Source | Notes |
|---|---|---|
| `hessian_snapshot` | `PeakOptionalityTracker` | H matrix at peak optionality. v18.9: tracker callable from external observation tooling. |
| `operator_snapshot` | `PeakOptionalityTracker` | Channel coverages + active-edge structure at peak optionality. |
| `causal_model` | (formerly FAO) | Admitted causal graph edges. **v18.9:** no longer written by FAO (FAO removed). Field remains for inheritance from prior runs. |
| `discovered_structure` | (formerly FAO) | Admitted axes from `ResidualExplainer` + `AxisAdmissionTest`. **v18.9:** no longer written. |
| `geometry_history` | (formerly `_log_geometry`) | Periodic snapshots of `{iter, phi, vol_opt, c_global}`. **v18.9:** no longer written by substrate. v19+ external tooling may write. |
| `semantic_attractors` | `ground_triplet` / `revise_triplet` | Basin records. |
| `basin_classification` | `promote_basin` / `demote_basin` | Subjective vs Objective per basin. |
| `agent_registry` | `_update_agent_registry` | Per-agent `{trust, trust_history, calls, last_seen}`; `trust_history` records `loop_closure` at each commit. v18.8: was `eog_history`. |
| `requester_frame` | (v18.1 seed) | Task framing seed. |
| `task_library` | (v18.1 seed) | Recurring task templates. |
| `output_format` | (v18.1 seed) | Agent response format constraints. |

**Removed in v18.7 (still absent):** `coupling_matrix`, `action_substrate_map`, `migration_history`.

**v18.9 status of FAO-written fields:** `causal_model`, `discovered_structure`, `geometry_history` are no longer updated by the substrate. They persist as inherited structure from prior runs and as forward slots for external tooling that may write to them. The fields are not removed from the dataclass — that would break inheritance from older ledger files.

---

## Architecture Constants (v19.0)

All values are explicit and substrate-internal — no hidden tunables, no globals beyond these.

### `SensingOperator`
```
INACTIVITY_THRESHOLD = 50           # iterations before channel deactivates
COVERAGE_DECAY       = 0.9          # per iter on absent signal
```

### `CompressionOperator`
```
HISTORY_WINDOW         = 10         # sensing_history depth used per iter
CONFIDENCE_INCREMENT   = 0.05       # on sign-agreement
CONFIDENCE_DECREMENT   = 0.025      # on sign-disagreement
INITIAL_CONFIDENCE     = 0.10       # for newly-proposed edges
CONFIDENCE_FLOOR       = 0.05       # below this, edge culled from active set
MAX_ACTIVE_EDGES       = 200        # cap on causal_graph size
MIN_DELTA_FOR_PROPOSE  = 0.05       # min motion magnitude to seed new edges
MIN_RESIDUAL_TRACKED   = 1e-3       # below this, residual ignored
NOISE_FLOOR            = 0.05       # default per-channel noise floor

alpha = clip(0.05 + 0.45 * surprise_ratio, 0.05, 0.5)  # surprise-derived
```

### `PredictionOperator`
```
UNIT_COMMIT_MAGNITUDE = 1.0         # hypothetical delta per affordance
```

### `CoherenceOperator`
```
EMA_ALPHA                = 0.05     # trajectory direction + loop signature EMA
DEVIATION_THRESHOLD      = 0.5      # signature_deviation gate for commit
STABLE_CHANNEL_THRESHOLD = 1e-3     # absolute floor for direction-bearing
                                     # also: 10% of strongest dir
```

### `SelfModifyingOperator`
```
MAX_DELTA = 1.0                     # per-channel clip for δf_rel
```

### `AgentRealityAdapter`
```
LATENCY_REFERENCE_S = 10.0          # latency channel scaling reference
TOKEN_REFERENCE     = 1000          # token-count channel scaling reference
UII_API_LLM_BUDGET  = 100000        # default token budget (env-overrideable)
```

### `AttractorMonitor`
```
CONFIDENCE_THRESHOLD = 0.5          # mean edge confidence to be Objective
VARIANCE_THRESHOLD   = 0.3          # max signature_deviation for Objective
VOL_OPT_THRESHOLD    = 1.0          # min vol_opt for Objective
```

### `observability.py` defaults (v19.0)
```
DEFAULT_BASELINE_WINDOW    = 30     # iterations of pre-perturbation history
DEFAULT_ENVELOPE_SIGMAS    = 3.0    # bounded-ness threshold (× baseline σ)
DEFAULT_METABOLIZE_HORIZON = 5      # iterations to expect decay-to-baseline
DEFAULT_RETURN_SIGMAS      = 1.0    # "returned to baseline" threshold (× σ)
DEFAULT_CONTINUITY_SIGMAS  = 5.0    # catastrophic-jump threshold (× σ)
```

These are tunable defaults, not derived from the math. The math says "bounded," "decays," "continuous"; these constants pick concrete thresholds. Calibration against an actual session may tighten them.

### Removed in v18.x — do not re-introduce as constants

- All CRK thresholds (`CRKMonitor` gone, v18.8)
- `_RANKING_STABLE_NEEDED`, `_MIN_ITERS_BEFORE_COMMIT` (EOG ranking gone, v18.8)
- `smo_snapshots_maxlen` (SMO `_snapshots` deque gone, v18.8)
- Any `C6_*` / `MAC_FCK_*` constants (load-bearing observe assertion gone, v18.8)
- `smo_epsilon = 0.1` as the global plasticity knob (compression's surprise-derived alpha is now the controlling scalar, v18.8)
- `PROVISIONAL_EVIDENCE_FLOOR`, `ADMITTED_EVIDENCE_FLOOR`, `PROVISIONAL_DECAY`, `ADMITTED_DECAY`, `SHORT_SESSION_THRESHOLD` (`ProvisionalAxisManager` gone, v18.9)
- `NONLINEAR_THRESHOLD`, `LAG_THRESHOLD`, `MIN_RECORDS_FOR_ANALYSIS`, `PREDICTIVE_GAIN_THRESHOLD`, `COMPRESSION_RATIO_THRESHOLD`, `SHUFFLE_CORRELATION_MAX`, `PHI_IMPROVEMENT_THRESHOLD`, `MIN_EVIDENCE` (`ResidualExplainer` / `AxisAdmissionTest` gone, v18.9)
- `_LOG_GEOMETRY_EVERY` (`_log_geometry` removed, v18.9)
- `_EPSILON_LOG` (geometry log removed, v18.9)

---

## Validation Signals (v19.0 — healthy run)

- `coherence.commit_decision` periodically returns an affordance name (the substrate is committing).
- `signature_deviation` sits below `DEVIATION_THRESHOLD = 0.5` for substantial spans (loop signature is stable; substrate is in regime).
- `trajectory_direction` has nonzero magnitude on multiple direction-bearing channels.
- Active channel count grows then plateaus as compression's `causal_graph` stabilizes; new channel arrivals on env perturbation extend the active set.
- `compression.causal_graph.edges` grows toward but doesn't sustain max-out at `MAX_ACTIVE_EDGES = 200` (edges form, prune; healthy churn).
- `surprise_ratio` sits in non-degenerate range (not 0, not 1 — substrate is neither saturated nor frozen).
- `smo.cumulative_delta` has nonzero entries on multiple channels (closure residuals re-entering sensing as relation signals).
- `self/*` channels populate with nonzero magnitude (Python introspection emitting; cross-class edges discoverable).
- **JSONL stdout stream emits one well-formed record per iteration**; no malformed lines, no gaps, no out-of-order iters.
- `delta_f_rel` is non-empty on most iterations (loop is producing motion; entries are bounded by `MAX_DELTA = 1.0`).
- `C_global` (computed externally by `uii_observer` or `observability`) is positive and slowly varying.
- Σ_P eigenvalues yield `vol_opt > 0` with periodic peak updates.
- Per-iter latency stays bounded (`self/clock/iter_latency` channel has stable trajectory_direction; no runaway).

### v19.0 observability-side validation

- `observability.Baseline` builds smoothly — non-degenerate `μ`, `σ`; 30-iteration window populates within first 30 iters of stream.
- Marked perturbations produce shape-analysis reads — bounded? metabolized? continuous? — with descriptive features rather than verdicts.
- Perturbations introduced through forward transducer (when implemented) show characteristic decay profile in `δf_rel` over the metabolize horizon.

---

## Warning Signals (v19.0 drift / architectural backsliding)

If any of these surface, the v19.0 architecture is not being preserved.

### Substrate drift
- `uii_observer` imported by `uii_geometry`, `uii_operators`, `uii_reality`, or `uii_ledger` — math spine has crossed back into the substrate.
- `uii_observer` imported by `uii_triad` — the v18.9 cleanup has been undone.
- `uii_fao.py` reintroduced, or `FailureAssimilationOperator`, `ResidualTracker`, `ResidualExplainer`, `AxisAdmissionTest`, `ProvisionalAxisManager` defined anywhere — v18.9 strip undone; parallel-curator scoring is back.
- `_log_geometry`, `_LOG_GEOMETRY_EVERY` reintroduced.
- Session_start / session_end ceremonial JSON events emitted from substrate.
- Per-commit or per-period status prints emitted to stdout (anything other than the JSONL record).
- Multiple stdout writes per iteration, or any JSON not matching `{iter, t, commit, delta_f_rel}` schema.
- `CRKMonitor` or `observe()` or `C1_*`/`.../C7_*` constraint identifiers — CRK has been re-introduced.
- Any of: `rank_actions_by_eog`, `expected_optionality_gain`, `score_actions`, `test_virtual`, `_RANKING_STABLE_NEEDED`, `_ranking_history`, `_last_ranking`, `_MIN_ITERS_BEFORE_COMMIT` — EOG scoring has been re-introduced.
- SMO with `_snapshots` deque, `revert_re_anchor`, `revert_recalc`, Repair / Maintain / Generate modes, or `predicted_delta` input parameter.
- Any operator's `apply()` taking more than one positional argument, or any operator reaching back through the loop to a downstream operator's state.
- `SymbolGroundingAdapter`, `ground_symbol`, `ground_trajectories`, or any `llm_client.call()` outside `AgentRealityAdapter.execute()`.
- `self_s`, `self_i`, `self_p`, `self_a` as feedback channels.
- `AffordanceClassRouter` or `get_viable_candidates`.
- `agent_registry` writing `eog_history` or any EOG-derived trust signal.
- Any `raise ProtocolExit`.
- Migration code in any form (`MigrationAttempt`, `migrate`, `migration_history`, `_should_migrate`, `boundary_pressure`, `DeathClock`).
- `CouplingMatrixEstimator` or any `coupling_estimator` parameter on any constructor.
- Hardcoded version strings (`'v18.X'`, `'v19.X'`) in log payloads.

### Observability drift (v19.0)
- `observability.py` or `cli_observe.py` importing anything from `uii_*` — JSONL-only access has been violated.
- `observability` querying operator state directly (Φ, ∇Φ, vol_opt, edge weights) — the library is no longer the doctor reading the patient's trajectory; it has opened the patient up.
- Shape-analysis output rendering verdicts ("coherent" / "failed" / "broken") rather than descriptive features ("stayed within envelope" / "returned to baseline at iter+3" / "no catastrophic jumps").
- The substrate emitting anything other than the `{iter, t, commit, delta_f_rel}` schema on stdout.

---

## Deprecated / Removed Concepts

These appear in older docs and earlier sub-versions but are not part of the v19.0 substrate. They are not coming back.

- Mortality / DeathClock / boundary_pressure / vital_capacity (Phase 7–9)
- Migration (Phase 7+; stripped v18.7)
- Six-phase `step()` (Phase 11; replaced by continuous `iterate()` in v17)
- Unified prompt at every commit (Phase 9; replaced by `GroundingSpec` at commit only in v16)
- `phi_legacy` / designer Φ formula (replaced v15.2 by geometry-derived Φ)
- `raise ProtocolExit` (added v18.1, fully removed v18.5)
- `SymbolGroundingAdapter` and `llm_client.call` outside `AgentRealityAdapter` (removed v18.4)
- `CRKMonitor` + C1–C7 firing logic + `repair_directives` (removed v18.8)
- EOG ranking commit pathway (removed v18.8)
- SMO snapshot reversibility (added v18.4, removed v18.8 — reversibility is compositional)
- Self-channels as scalar feedback (added v18.0, replaced v18.8 by Python introspection)
- `AffordanceClassRouter` / three-class candidate workflow (added v18.0, removed v18.8)
- `CouplingMatrixEstimator` + `coupling_estimator` parameter cascade (removed v18.7)
- Migration code + `MigrationAttempt` + `migration_history` (removed v18.7)
- `uii_fao.py` and the entire FAO residual-explainer / axis-admission pipeline (removed v18.9)
- `_log_geometry` and the substrate-side math-spine logging path (removed v18.9)
- Session ceremonial events and per-commit status prints (removed v18.9)

---

## Canonical Reminders (v19.0)

If unsure about how to extend the substrate, return to these.

1. **The substrate contains no scoring inside it.** The math spine is descriptive — it reads emitted state from outside the loop. Any path that has the substrate consult its own scalar quality measure to adjust behaviour is incoherent with the architecture.
2. **Coherence is the gate.** There is no other commit pathway. `coherence.commit_decision` is consulted directly by `iterate()`; `None` is a valid output that means "hold this iteration."
3. **SMO is a closure-residual emitter, not a state-patcher.** It computes `δf_rel` from coherence + prediction's projections, clips, accumulates, and emits `relation_signals`. It does not revert, repair, or maintain.
4. **Strict I/O is non-negotiable.** Each operator's `apply()` takes one positional argument: the prior operator's output. Carriers propagate forward. No operator reaches back.
5. **Sensing is the integrator.** Probing happens once per iteration before sensing runs. `env_signal` accumulates from probe + self channels + affordance channels + ledger channels + previous-iter SMO `relation_signals`. Sensing reads `env_signal` and produces the perceptual surface — it is not pulled from anywhere. *(Phase 14 commitment: forward transducers extend `env_signal` with external-input channels under the same I/O contract.)*
6. **Affordances are channels, not categories.** Each affordance becomes a channel through `_build_affordance_channel` with availability/just-used signal. Compression's normal co-movement detection learns affordance→outcome edges as ordinary causal sources.
7. **`AgentRealityAdapter` is the only LLM gateway.** There is no `llm_client.call()` elsewhere. LLM perturbations enter as channel readings; subsequent actions consume responses only after they have cycled through SIPA.
8. **Reversibility is compositional.** On no-commit, the world has not moved; on commit, it has, and next iteration's sensing metabolizes the consequences. There is no rollback mechanism.
9. **Trust is observational.** `agent_registry` records `coherence.consistency.loop_closure` at moments of agent commit. The substrate does not select agents to maximize trust.
10. **C_global is positive by construction.** Instantiated systems are in-protocol per Spine §11. The substrate's coherence is asserted at instantiation; whether the running loop is in regime is what an external observer reads from the trajectory.
11. **The triadic mapping is structural.** `T(x) = f_rel(f_self(x), f_env(x))` is what the closure loop computes on every iteration through compression's edge formation across self / env / affordance / relation channels — not a math-spine constraint applied externally.
12. **The substrate's only output is its causality.** *(v18.9)* Every `iterate()` ends with one `{iter, t, commit, delta_f_rel}` line on stdout. `commit` is the cause the substrate pushed; `delta_f_rel` is ẋ — the per-channel motion that resulted, in the math spine's coordinate frame. The schema isn't a design choice; it's what the architecture's invariants permit. No banners, no summaries, no math-spine readings, no ceremonial events — anything else would re-introduce content the substrate is forbidden from producing. Readers external to the loop interpret. stderr is for execution faults only.
13. **Observation is JSONL-only.** *(v19.0)* `observability.py` and `cli_observe.py` import nothing from `uii_*`. They read the JSONL stream and report descriptive features of trajectory shape. Verdicts ("coherent" / "failed") are out of scope; the user reads the gestalt.
14. **Shape-mismatch is symmetric.** *(v19.0)* A divergence between predicted and observed shape could mean closure is breaking *or* the prediction was bad. The library reports features; interpretation is the user's work.
15. **Forward transducers carry external perturbation.** *(v19.0 architectural target)* External input enters the substrate the same way every other channel does — through `env_signal`. Discontinuous perturbation, on the perturber's schedule, with no special-case reception path.

---

## Evolution Cycle

The pattern across the project's history, restated for v19.0:

1. New architectural commitment surfaces (e.g., "the LLM is substrate not tool," "the math spine is descriptive not internal," "the substrate's only output is its causality," "the substrate has interface surfaces").
2. Sub-versions implement pieces of the commitment while still standing on prior scaffolding.
3. The gap between commitment and implementation surfaces as named structural failures.
4. A version (v18.8 for closure structure, v18.9 for emission, v19.0 for interface surfaces) rebuilds the substrate around the commitment, removing the scaffolding.
5. Constants get re-derived structurally rather than chosen by hand.
6. Canonical reminders are rewritten to match the new architecture.

**The commitment for Phase 14 is:** the substrate has explicit interface surfaces. External entities can perturb it discontinuously through forward transducers; external readers can interpret its trajectory through pure JSONL consumers. The substrate's strict-I/O contract is preserved; what changes is the framing of *what's outside* — Phase 13 said "LLMs are substrate"; Phase 14 says "and substrates have interface surfaces."

**The math spine becomes a public artifact — one structural line across three versions.** v15.2 derived Φ from `CompressionOperator` geometry rather than a designer formula: the math spine becomes *descriptively correct* against the substrate it describes. v18.8 externalized the spine entirely — the substrate stopped reading its own scalars; Φ, ∇Φ, vol_opt, C_local moved to `uii_observer` and became things computed *about* the substrate, not *by* it. v18.9 made the substrate emit ẋ as one record per iteration: the spine's central dynamical object — `ẋ = P_M(G⁻¹ · ∇Φ)` — is now a public artifact anyone with the stream and the spine can read against. From private description, to externally positioned reader, to publicly emitted content. Each step moved the math spine's central object closer to being readable from outside; v18.9 is the version where it actually became readable. v19.0's external readers (`uii_observer` against operator state, `observability` against the JSONL stream) read what was, mathematically, already the right thing to read.

**Simplifications come from visibility, not invention.** The v18.9 strips — FAO removed, ceremonial events removed, math-spine logging removed — were not architectural breakthroughs. While FAO was inside the substrate, it was doing the work of curating which residuals counted as axes; there was nowhere else for that work to live, so it didn't read as misplaced. Once `uii_observer` was external (v18.8) and the substrate's emission was constrained to its causality (v18.9), FAO's parallel-curator role became visible as a re-entry of scoring into the substrate. The pattern: scaffolding only becomes visible *as scaffolding* once the load-bearing structure it was supporting is actually present.

**v19.0 is "bones."** Forward transducers are gestured at, not built. Persistence is graceful-Ctrl+C, not crash-safe. The schema between substrate emission and observability ingestion needs reconciling. Calibration of the shape-prediction defaults is empirical work for v19.x. But the architecture is in the right shape — substrate emits, external readers interpret, external perturbers can push input through the existing sensing pathway when forward transducers are built.

The base is solid; Phase 14 builds off it.

---

## Trajectory Open Issues (carry into v19.1+)

### Schema reconciliation between substrate emission and observability ingestion

`uii_triad.iterate()` emits `delta_f_rel` as `Dict[str, float]` (per-channel non-zero entries). `observability.JSONLRecord.from_line` parses it as `float` via `float(d['delta_f_rel'])`. The dataclass shape is wrong for the emitted schema; on a real substrate stream the parser will fail.

Two reconciliation paths:
1. Observability ingests the dict and reduces it to a scalar (e.g., L2 norm, mean abs, max abs) before applying baseline / envelope / continuity tests. Preserves substrate emission shape; observability picks the reduction.
2. Substrate emits a scalar reduction alongside or instead of the per-channel dict. Compresses the emission; loses per-channel resolution at the emission interface.

Path 1 is the architecturally cleaner choice — substrate emits its full structural state, readers reduce as their analysis needs. Path 2 reads as scoring-at-emission and is suspect.

### Forward-transducer adapter not yet built

The Phase 14 architectural commitment is that external input enters the substrate through the existing `env_signal` accumulation pathway. v19.0 does not contain the running adapter. The README sketches an integration shape (`triad.user_reality.push_input(...)`); the actual `UserRealityAdapter` (or whatever name lands) is v19.1 work. Until built, the substrate's `env_signal` still accumulates only from Browser + Agent + self/affordance/ledger introspection.

### Persistence is graceful-Ctrl+C only

A crash still loses post-last-save deltas. Inheriting a partial state across crashes — or making the JSONL stream itself the durable record from which substrate state can be reconstructed — is a v19.x design question. The current ledger-on-Ctrl+C path is correct as far as it goes; it just doesn't go far enough for systems that crash.

### Self / Relation affordance reachability gap

`PredictionOperator` projects only `BASE_AFFORDANCES`. `SELF_AFFORDANCES` (`read_ledger`, `write`, `write_outreach`) and `RELATION_AFFORDANCES` (`ground_triplet`, `revise_triplet`, `promote_basin`, `demote_basin`) are routable through `CompositeRealityAdapter.execute` but currently invisible to coherence's commit pathway. The execute branches are dead unless something else triggers them. Resolution likely involves either widening the projection set or adding class-specific commit pathways. Open since v18.8.

### `viable_action_types` population path

`iterate()`'s commit dispatch checks `current_affordances.get('viable_action_types', set())` against `commit_decision`. The browser adapter populates this with link/button/input action types; the agent and self-routing branches don't currently surface their executable affordances into `viable_action_types`. Verify the dispatch isn't silently skipping committed `query_agent` / `read_ledger` / etc. Open since v18.8.

### `llm_query` / `query_agent` consolidation

`BASE_AFFORDANCES` contains both names. `query_agent` is the executable handler; `llm_query` has no handler. Either consolidate or split into distinct executable paths. Open since v18.8.

### `write_outreach` implementation

Currently logged-only — no comm-channel mechanism. Affordance is sense-able as a channel, routable through `_execute_self`, but the actual write is unimplemented. Open since v18.2.

### Calibration of observability shape-prediction defaults

`DEFAULT_BASELINE_WINDOW = 30`, `DEFAULT_ENVELOPE_SIGMAS = 3.0`, `DEFAULT_METABOLIZE_HORIZON = 5`, `DEFAULT_RETURN_SIGMAS = 1.0`, `DEFAULT_CONTINUITY_SIGMAS = 5.0` are first-pass values. The math says "bounded," "decays," "continuous" qualitatively; these constants pick concrete thresholds. Calibration against real sessions is empirical work for v19.x.

### Long-run `loop_closure` validation as trust signal

Replacing EOG-history with `loop_closure`-history in `agent_registry` has the right architectural shape, but whether `loop_closure` tracks meaningful agent-quality differences across realistic agent populations is empirical. Open since v18.8.

### Σ_P from emission as a design commitment

`uii_observer._build_sigma_p` constructs Σ_P from compression's `causal_graph` + sensing's coverage. The change has the right architectural shape (Σ_P is observational, not a substrate computation), but the calibration of `vol_opt` against the v17 baseline is open. Open since v18.8.

---

## Version Control Test

Before declaring a code change is correct against v19.0, all of the following must hold.

### v18.8 invariants (still required)

1. `uii_observer.py` exists. The substrate's modules contain no symbol named `PhiField`, `eigen_decompose`, or any `compute_c_local` logic. `uii_geometry.StateTrace` is a stub.
2. No symbol named `CRKMonitor`, `observe` (as method on a constraint checker), `evaluate_pre_action`, `repair_directives`, or `_CRK_OBSERVE_CONSTRAINTS_EVALUATED` anywhere in the codebase.
3. `coherence.commit_decision` is the only commit gate. `iterate()` has no other call to anything resembling "select committed action."
4. No symbol named `rank_actions_by_eog`, `expected_optionality_gain`, `score_actions`, `test_virtual`, `_RANKING_STABLE_NEEDED`, `_ranking_history`, `_last_ranking`, or `_MIN_ITERS_BEFORE_COMMIT`.
5. `SelfModifyingOperator` has no `_snapshots` deque, no `revert_re_anchor` / `revert_recalc` methods, no Repair / Maintain / Generate mode branches, no `recent_external_attribution` flag.
6. `SelfModifyingOperator.apply` takes exactly one positional argument: `coherence`. No `predicted_delta` parameter.
7. Each operator's `apply()` takes exactly one positional argument that is the prior operator's output. `iterate()`'s body matches the strict-I/O wiring.
8. `AgentRealityAdapter.execute` is the only call site of `llm_client.call()`. No `SymbolGroundingAdapter`, `ground_symbol`, `ground_trajectories` anywhere.
9. `AffordanceClassRouter` is absent. Affordances enter as channels through `_build_affordance_channel`.
10. `self_s`, `self_i`, `self_p`, `self_a` do not appear as channel ids. Self channels have `self/...` hierarchical names.
11. `agent_registry` records `trust_history` (loop_closure values) and not `eog_history`.
12. No `raise ProtocolExit` statement, no `ProtocolExit` class, no `except ProtocolExit` branch.
13. No `MigrationAttempt`, `migrate` action_type, `migration_history` field, `_should_migrate` method.
14. No `CouplingMatrixEstimator` class, no `coupling_estimator` constructor parameter on any class, no `coupling_matrix` or `action_substrate_map` ledger writes.
15. Hardcoded version strings (`'v18.0'`, `'v17'`, etc.) are absent from log payloads.

### v18.9 invariants (additional)

16. **`uii_fao.py` does not exist** in the repository. No `FailureAssimilationOperator`, `ResidualTracker`, `ResidualExplainer`, `AxisAdmissionTest`, or `ProvisionalAxisManager` class definitions anywhere.
17. **`uii_observer` is not imported by any substrate file.** Specifically: `uii_geometry`, `uii_operators`, `uii_reality`, `uii_ledger`, `uii_triad` import nothing from `uii_observer`.
18. **`_log_geometry` does not exist** as a method on `MentatTriad`. No `_LOG_GEOMETRY_EVERY` constant.
19. **`iterate()` ends with exactly one `sys.stdout.write(json.dumps({...}))` call** with the schema `{iter, t, commit, delta_f_rel}` and a trailing `\n` and `sys.stdout.flush()`. No other stdout writes anywhere in the substrate code path.
20. No session_start / session_end ceremonial JSON events emitted from the substrate.
21. No per-commit, per-period, or per-N-iteration status prints from the substrate (status output, if any, goes through stderr — and even then only for execution faults).
22. `FAO.distill_to_ledger` is not called in any `finally` block. The `__main__` finally calls `ledger.save_ledger()` only.

### v19.0 invariants (additional)

23. `observability.py`, `cli_observe.py`, and the v19.0 `README.md` exist as separate files.
24. **`observability.py` and `cli_observe.py` import nothing from `uii_*`.** Specifically: no `from uii_geometry`, `from uii_operators`, `from uii_reality`, `from uii_ledger`, `from uii_observer`, `from uii_triad` anywhere in the v19.0 utility files.
25. `observability.JSONLRecord` accepts the substrate's emitted schema (or includes a documented reduction of `delta_f_rel` from dict to scalar — the schema reconciliation may land in v19.1).
26. Shape-analysis output in `observability` and `cli_observe` is descriptive — features like "stayed within envelope," "returned to baseline at iter+3," "no catastrophic jumps." No verdicts ("coherent" / "broken" / "failed").

**No to any of these = do not commit.**
