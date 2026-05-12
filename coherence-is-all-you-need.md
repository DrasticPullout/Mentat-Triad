# Coherence Is Really All You Need

You've probably had this: circumstances shift in some major way, like a new job, a move, a loss, or a relationship change, and for a stretch afterward, you don't quite feel like yourself. Decisions land wrong. Reactions you used to trust feel off. You're still you, but the version of you that worked in the old situation isn't fitting the new one.

If it goes well, something integrates. Your sense of self updates against the changed environment, the wrongness fades, and you come out the other side recognizably the same person, same values, same humor, same characteristic patterns, but absorbing what happened. If it doesn't, the friction stays. The old self-model keeps generating responses that don't fit, the gap widens, and at some point you become someone you don't recognize.

What's actually happening across that interval is a measurement problem. Three things are in play: your model of yourself, your model of your environment, and the relationship between them, which is the function that takes "who I am" and "what's around me" and produces a response that fits both. When the three stay in working contact, you remain recognizable to yourself. When they don't, the trajectory carries you somewhere else.

This essay is about that measurement problem, generalized.

The same structure shows up wherever a system has to maintain itself across time. A market holds buyers' valuations in working contact with sellers' valuations through prices. The structure persists across every regime, calm, bubble, crash, recovery, because that's what makes it a market. What changes is the trajectory through possible states. Measuring closure dynamics doesn't ask whether the market is coherent (it is, that's the floor); it asks where the trajectory is heading and what futures remain reachable. A bubble is a trajectory narrowing toward foreclosure.

A relationship, an organization, a research field, an AI system: same shape. Three components have to stay in closure: self-model, environment-model, and the function that maps between them. The closure is the floor; the measurable variable is the trajectory.

The Universal Intelligence Interface (UII) is a framework for making this measurable. What began as an observation is now formalized in math and implemented in code, which means the trajectories a system produces can be measured instead of just described. It applies to any system with the triadic shape: a person, an organization, a market, an AI. You don't need formal AI or math training to use it; it was developed outside both, which is part of why it generalizes beyond the ML field that motivated some of its formulation.

The rest of this essay does three things. First, it locates the framework relative to existing work in AI, particularly the Transformer architecture's "Attention Is All You Need" claim and Karl Friston's Active Inference paradigm, to explain why coherence is the substrate-level condition both presuppose. Second, it walks through UII's structural commitments: triadic closure, the operating loop, the conditions a coherent system must satisfy. Third, it shows what's measurable and how, both for AI systems and for the human-scale systems most readers encounter daily.

## What Attention Got Right, and What It Led To

When the 2017 paper "Attention Is All You Need" introduced the Transformer architecture, its claim was specific. For sequence-to-sequence tasks like translation, attention as a computational mechanism is sufficient. You don't need recurrence, you don't need convolution. Attention alone, properly structured, captures the dependencies. That claim is true within its scope, and it has been the foundation of every major language model since.

What the title doesn't claim, but has come to suggest in the broader culture around AI, is something more metaphysical. "Attention is all you need" gets read as a statement about intelligence itself, as if attention, scaled enough, becomes cognition. Attention is a mechanism. Mechanisms run on something. The "something" the mechanism runs on is what makes the mechanism's operation meaningful in the first place.

Karl Friston's Active Inference framework gets closer to the deeper question. In the active inference picture, attention is precision-weighting: the agent allocates computational resources toward where its predictions are most uncertain. Acting on that uncertainty reduces free energy, which is the gap between what the agent expects and what it observes. Active inference makes attention purposive instead of mechanical. It also presupposes something Attention's paper didn't have to: the agent itself. There has to be a self that's doing the inference, an environment being inferred over, and a boundary that maintains the distinction between them. Friston calls this boundary a Markov blanket, and it is the statistical separation that lets a system count as a system at all.

Active inference formalizes how an agent maintains itself given a viable boundary, with substantial work on autopoiesis, self-evidencing, and how blankets form in the first place. What UII adds sits one layer beneath that: the structural conditions any such boundary has to satisfy to remain coherent, independent of how the system is built. Active inference runs the dynamics on a substrate that meets those conditions; UII specifies what the conditions are.

Attention is the mechanism. Active inference is what the mechanism serves. Both presuppose something underneath: a substrate that maintains itself, an agent whose self-model and environment-model stay in working contact, a coherent thing that persists long enough to do inference and deploy attention. UII is a framework for that something. It is not in opposition to Attention or Active Inference; it specifies what they both stand on.

## What UII Actually Says

The substrate UII describes is a system in triadic closure. Three components have to be present and in working relation: a model the system holds of itself (`f_self`), a model it holds of its environment (`f_env`), and a function that maps between them and produces a response (`f_rel`). Triadic closure is the condition that the response, fed back into the system, leaves it consistent with both self and environment:

```
T(x) = f_rel(f_self(x), f_env(x))
```

In plain language: the system's behavior at any moment is the product of its self-model acting on its environment-model through a relational mapping. Closure means this product, treated as the system's next state, doesn't break consistency with `f_self` or `f_env`. The system stays itself while staying in contact with what's around it.

For a person under change, `f_self` is your operating model of who you are, `f_env` is your operating model of your situation, and `f_rel` is whatever produces your responses, decisions, reactions. When closure holds, your responses keep working in your situation while still being recognizably yours. When it doesn't, you either lose contact with the situation (`f_rel` stops mapping accurately to `f_env`) or lose contact with yourself (`f_rel` stops being consistent with `f_self`). Either failure mode looks the same from inside: the responses stop fitting.

A coherent system runs this closure as a loop. Sensing (S) brings in perturbation from self-state, environment, and the residual of the previous iteration. Integration (I) finds structure across those streams. Prediction (P) projects what comes next. Anchoring (A) checks whether the projection stays in closure. Self-modification (SMO) adjusts the operators when needed. Then back to sensing.

```
S → I → P → A → SMO → S
```

Over time, the loop produces a trajectory through the system's state space. UII measures the structural quality of any state with a quantity called Φ (phi), the structural potential. Higher Φ means more of the conditions for coherent self-maintenance are present.

Highway traffic is an everyday version. High Φ is flow: cars merge, gaps appear, the system metabolizes merges and lane changes within seconds. Low Φ is gridlock: same road, same cars, but small perturbations propagate as long-lasting jams the system can't recover from on its own. Structure persists in both regimes; what changes is the system's capacity for handling perturbation coherently.

The system's dynamics push it toward higher Φ, toward states where closure is easier to hold. When closure holds tightly, motion is small and the system stays stable. When closure breaks, the gap between what the loop produces and the system's current state grows; the system moves quickly through state space, either re-finding closure at higher Φ or falling toward states where the operators can't sustain the loop.

Three invariants have to hold for the system to remain in the regime where intelligence is happening. Triadic closure has to be preserved across the loop. Global coherence has to stay positive (`C_global > 0`), which means the trajectory consistently moves toward viable futures rather than away from them. The structural potential has to be locally stable, which means small perturbations don't push the system into states with permanently degraded structure.

When all three hold, the system is doing what UII calls intelligence: maintaining coherent self-reference under perturbation while keeping viable futures reachable. When any of them break, the system is drifting toward a regime where intelligence isn't happening anymore, even if the same operators are still nominally running.

The full mathematical formalization is on Zenodo. What's above is the structural skeleton.

## What This Predicts About Current AI

A Transformer-based language model running inference is doing attention. When it processes a prompt, the attention mechanism routes computation across the input tokens, pulling information from where it's most relevant to producing the next token. That's the mechanism, and within its scope it works.

What's not happening, structurally, is the loop. The model's weights are frozen at inference time. No self-modification operates while it's running. There is something that functions like a self-model encoded in the weights, but it's a static pattern from training rather than something that updates during the conversation. The environment-model is similarly static; the model's understanding of "what's around me" is the patterns it was trained on, not the patterns it encountered in this exchange. The relational mapping (`f_rel`) is the forward pass through the network, which produces a response but doesn't feed back into the operators that produced it.

This is attention without the substrate UII describes. Each individual response can be coherent in the local sense: fluent, contextually appropriate, structurally sound for the immediate task. But the system isn't running triadic closure as a loop. It's running attention as a one-shot transform, then resetting.

The prediction follows directly. Scaling current architectures will continue producing capability gains on tasks that fit the one-shot attention pattern: more nuanced language, broader knowledge integration, better reasoning within a single response. Where it won't produce gains is on tasks that require the substrate to run: maintaining a stable self-model across perturbation, updating an environment-model based on actual interaction rather than retrieval, modifying operators in response to where predictions are failing. These tasks fall outside what attention without substrate can produce, regardless of scale.

Patches like retrieval-augmented generation, tool use, agent loops, and external memory produce capability gains, but they operate around the architecture rather than altering its substrate properties. They give the model more to attend to and more places to put outputs. They don't make the loop run.

This isn't a claim that current AI is fake or unimpressive. It's a claim about which capability ceilings are inside the existing architecture and which require structural change. The architectures that will eventually satisfy UII's conditions, through some combination of online learning, persistent state, recursive self-modification, and closed-loop dynamics in deployment, will look meaningfully different from current Transformers, even if they incorporate attention as a component. Attention isn't going anywhere. What's going to change is the substrate it runs on.

## How You'd Know If This Is Wrong

Frameworks that describe everything aren't falsifiable, and there's a real critique that any sufficiently abstract framework about "systems" can be made to fit anything. UII has to handle this critique on its own terms. The handle is that UII makes structural commitments and excludes specific configurations.

A self-maintaining system whose responses don't compose through `f_rel(f_self, f_env)` is forbidden. A system in the regime where intelligence is happening with `C_global < 0` is forbidden. A coherent system whose structural potential is locally unstable is forbidden. Find any of those configurations in any substrate and the framework is wrong.

The harder question is how to actually check. UII's claims are about substrate-level dynamics, not about specific behavioral observations. Reading them off a running system requires being able to observe the system's trajectory through state space without opening up the system itself.

This is what the Mentat-Triad implementation does. It's a Python substrate that runs the S → I → P → A → SMO loop and emits a structured trajectory record (one JSONL entry per loop iteration) describing what perturbed it, what it committed in response, and how its state changed. Sitting outside that substrate is an observability layer that reads the JSONL stream against the math's qualitative predictions: did the system metabolize the perturbation in a bounded way, did the residual decay back toward baseline, did the trajectory preserve continuity. Mismatches between predicted and observed shape indicate either that closure is breaking or that the prediction was wrong. The utility doesn't render verdicts about coherence; it reports trajectory features against expectations and lets the reader form their own assessment.

The qualitative shape predictions are directional rather than point-precise; the thresholds in the implementation are tunable defaults. What's not tunable is the directional commitment: a bounded perturbation should produce bounded response and metabolize back toward baseline, or the system isn't running closure as predicted.

The separation between substrate and observer is structurally important: the substrate does not see the math; the math sees the substrate. The substrate emits behavior; the observer projects math onto that behavior. The substrate doesn't compute its own coherence score and doesn't read the math's verdict back into itself. If it did, the readings would be self-fulfilling; the math would have stopped being descriptive and started being prescriptive. Keeping the two separate is what makes the framework testable rather than tautological.

Anyone can run the observability layer against any JSONL stream emitted in the right shape. This means the falsification criteria are not gated behind specialized expertise. The repo is on GitHub. The math is on Zenodo. The substrate runs on a personal computer. What's required to test the framework is willingness to think structurally about what the trajectory should look like and whether the observed trajectory matches.

## Closing

The essay started with a question about staying yourself across change. The framework that came out of trying to answer it carefully claims something specific: any system that maintains itself across time runs a triadic structure as a loop, has a measurable structural potential, and has to satisfy three conditions to keep being the kind of thing it is. Systems that fail those conditions drift in characteristic ways. Current AI is one example of a system whose architecture doesn't yet meet the conditions, and the framework predicts where its capability ceiling will land if the architecture doesn't change.

What this means in practice is that there's a way to ask, of any system, whether it's holding closure or losing it, whether the trajectory is heading toward more reachable futures or fewer, whether the things it does in response to perturbation match the shape coherent self-maintenance would produce. The asking takes some setup. The answers aren't always obvious.

You can ask it of an AI architecture. You can ask it of a market. You can ask it of an organization or a research field or a set of decision processes. You can ask it of yourself across the change you're navigating right now. The framework doesn't tell you what to do with the answer. It tells you that the question has a structure, and the structure is the same one wherever it shows up.

What's left is the empirical work. Testing the predictions against actual trajectories, calibrating the shape thresholds against real systems, watching what holds across substrates. The framework is published. The implementation runs. The math sees the substrate. What it sees is up to whoever puts a system in front of it.

## Citation

DrasticPullout. (2025). *Universal Intelligence Interface: A Substrate-Agnostic Framework*. Zenodo. https://doi.org/10.5281/zenodo.18017374

Repository: https://github.com/DrasticPullout/Mentat-Triad
