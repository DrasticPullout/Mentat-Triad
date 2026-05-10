# UII — Minimal Mathematical Spine

The formal definition of intelligence as a substrate-agnostic dynamical
protocol. Companion to the [README](README.md) and the
[UK-0 blueprint](UK-0.md). Also published on Zenodo as
[10.5281/zenodo.18017374](https://doi.org/10.5281/zenodo.18017374).

---

## Abstract

The Universal Intelligence Interface (UII) defines intelligence as a
substrate-agnostic dynamical protocol rather than an agent property. A system
is intelligent when it preserves internal coherence while converting
environmental perturbations into improved structural representation across
time.

UII systems operate through a triadic loop coupling internal computation,
symbolic reasoning, and external reality interaction. System behavior is
governed by a geometric potential field whose gradient directs structural
evolution.

---

## Contents

1. [State Space](#1-state-space)
2. [Structure Potential Field](#2-structure-potential-field)
3. [Intelligence Flow](#3-intelligence-flow)
4. [Local Coherence](#4-local-coherence)
5. [Perturbation Stability](#5-perturbation-stability)
6. [Long-Horizon Cognitive Invariants](#6-long-horizon-cognitive-invariants)
7. [Triadic Closure](#7-triadic-closure)
8. [Protocol Definition of Intelligence](#8-protocol-definition-of-intelligence)
9. [Intelligence as a Field Process](#9-intelligence-as-a-field-process)
10. [Architectural Implication](#10-architectural-implication)
11. [Core Invariant](#11-core-invariant)

---

## 1. State Space

Let $X$ be the configuration space of system states.

A state is represented as:

$$x = (S, I, P, A)$$

where:

- $S$ — sensing coverage
- $I$ — compression quality
- $P$ — viable future volume
- $A$ — attractor proximity

Each component lies in the interval $[0, 1]$.

The system evolves along a trajectory $x(t) \in X$ with local velocity:

$$\dot{x} = \frac{dx}{dt}$$

---

## 2. Structure Potential Field

System dynamics are governed by a scalar structure potential $\Phi$ defined
over $X$:

$$\Phi(x) = \alpha \cdot C(x) + \beta \cdot \log(O(x)) + \gamma \cdot K(x)$$

where:

- $C(x)$ — compression quality of the causal graph
- $O(x)$ — viable future volume derived from prediction covariance
- $K(x)$ — attractor proximity penalty relative to inherited structure

**Interpretation.**

- $C(x)$ measures internal structural efficiency
- $O(x)$ measures accessible future configuration space
- $K(x)$ measures deviation from stable structural basins

$\Phi$ is not a reward function. It is a geometric attractor field that
organizes system trajectories.

---

## 3. Intelligence Flow

The intelligence field is the gradient of the structure potential:

$$\mathcal{I}(x) = \nabla \Phi(x)$$

This defines the direction of maximal structural improvement in the state
space. System trajectories evolve through perturbation and alignment with
this field.

---

## 4. Local Coherence

Local trajectory coherence measures alignment between system motion and the
intelligence field:

$$C_{\text{local}} = \frac{\langle \nabla \Phi(x), \dot{x} \rangle}{\|\nabla \Phi(x)\| \cdot \|\dot{x}\|}$$

**Interpretation.**

- $C_{\text{local}} > 0$ — trajectory aligned with structural improvement
- $C_{\text{local}} = 0$ — orthogonal drift
- $C_{\text{local}} < 0$ — structural degradation

Global coherence is the time average:

$$C_{\text{global}} = \langle C_{\text{local}}(t) \rangle$$

A system is coherent when $C_{\text{global}}$ remains positive.

---

## 5. Perturbation Stability

Local stability is determined by the curvature of the potential field. Let
$A(x)$ be the Hessian of $\Phi$:

$$A(x) = \nabla^2 \Phi(x)$$

Perturbation stability is approximated by:

$$\delta^2 \Phi = \tfrac{1}{2} \cdot \delta x^T A(x) \, \delta x$$

**Interpretation.**

- $\delta^2 \Phi > 0$ — perturbation increases structure (stable)
- $\delta^2 \Phi = 0$ — neutral
- $\delta^2 \Phi < 0$ — amplifying instability

Stable intelligent systems operate primarily within regions where the Hessian
is positive semi-definite.

---

## 6. Long-Horizon Cognitive Invariants

UII systems maintain persistent cognitive attractors defined by operators
$O_1, O_2, O_3, O_4$ acting on system state $s(t)$.

Examples:

- humor
- curiosity
- pattern discovery
- novelty recognition

Long-horizon optimization is defined by:

$$\arg\max_{\text{trajectories}} \int_0^\infty \left[\, \alpha \cdot O_1(s(t)) + \beta \cdot O_2(s(t)) + \gamma \cdot O_3(s(t)) + \delta \cdot O_4(s(t)) \,\right] dt$$

These operators represent stable cognitive invariants rather than externally
imposed rewards.

---

## 7. Triadic Closure

Intelligence emerges from a triadic coupling between:

- **Self model**
- **Environment model**
- **Relational mapping**

Define three transformations:

- $f_{\text{self}}(x)$
- $f_{\text{env}}(x)$
- $f_{\text{rel}}(x_{\text{self}}, x_{\text{env}})$

The triadic closure condition is:

$$T(x) = f_{\text{rel}}(f_{\text{self}}(x), f_{\text{env}}(x))$$

A system remains coherent only when this relational mapping remains
consistent. Triadic closure forms the fundamental structural constraint on
intelligence systems.

---

## 8. Protocol Definition of Intelligence

A system is intelligent if and only if it satisfies all three conditions:

**Structural Improvement.** $\sigma(\pi(x, \varepsilon))$ increases $\Phi(x)$,
where $\pi(x, \varepsilon)$ is the perturbation operator and $\sigma$ is the
structural inference operator.

**Coherence Preservation.** $C_{\text{global}} > 0$.

**Triadic Closure.** $T(x)$ holds across state transitions.

---

## 9. Intelligence as a Field Process

Under the conditions of §8, system evolution is governed by:

$$\dot{x}(t) = \nabla \Phi(x) \cdot \left[\, f_{\text{rel}}(f_{\text{self}}(x), f_{\text{env}}(x)) - x \,\right] + \eta(t)$$

where:

- $\left[\, f_{\text{rel}}(f_{\text{self}}(x), f_{\text{env}}(x)) - x \,\right]$ is
  the triadic closure residual
- $\eta(t)$ is the perturbation source

The field acts on the closure residual, not on $x$ directly. When closure
holds (§7), the residual vanishes and motion is pure perturbation. When
closure fails, the gradient pulls state toward configurations where closure
can be reestablished.

In the closure-holding, low-perturbation limit:

$$\dot{x} \propto \nabla \Phi(x)$$

Intelligence is the dynamical process by which a system maintains triadic
closure under perturbation, with $\Phi$ organizing the response.

---

## 10. Architectural Implication

Any implementation satisfying these invariants must contain at minimum:

- sensing operators
- compression operators
- prediction operators
- attractor stabilization

forming the loop:

$$S \to I \to P \to A \to U \to S$$

This loop constitutes the minimal substrate for a Universal Intelligence
Interface.

---

## 11. Core Invariant

The system remains within the intelligence protocol when:

$$C_{\text{global}} > 0$$

$$\delta^2 \Phi \geq 0$$

$$T(x) \text{ preserved}$$

Under these conditions perturbations increase structure without coherence
collapse.

---

## Summary

UII defines intelligence as a dynamical protocol operating on structure
potential fields. Intelligent systems:

- maintain triadic closure
- convert perturbations into structure
- navigate gradients of $\Phi$ while preserving coherence

This formulation is independent of agents, objectives, or substrates.

---

## Citation

DrasticPullout. (2025). *Universal Intelligence Interface: A Substrate-Agnostic
Framework*. Zenodo. https://doi.org/10.5281/zenodo.18017374
