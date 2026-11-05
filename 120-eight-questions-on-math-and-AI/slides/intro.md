# Eight Big Questions About AI

## A discussion, not a verdict

* Webinar — November 6, 2026

* Mark Kerzner

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/title.png (any filename is fine).

Framing to say out loud at the start: these are eight questions I am still working through, not eight positions I hold. Each slide carries one cue and one prompt; the thinking is in these notes.

Attribution caution: this list is ours. It was prompted by watching the Terence Tao video linked in the project README ("AI Is Moving Faster Than We Can Control"), but nothing here should be presented as his claim, his view, or his proof. The only Tao source cited anywhere in this deck is his article "Machine-Assisted Proof", Notices of the AMS 72, no. 1 (DOI 10.1090/noti3041) — and the month of that issue was not confirmed, so say "Notices, volume 72" rather than a date.

Thread that runs through the whole hour: the thing I am most worried about losing is mathematical discovery — new concepts, connections and methods. Not student practice, not the enjoyment of solving problems. Questions 4, 5 and 8 all come back to it.

How to use the deck: the slide is the memory cue. Objections are in the notes under each question, deliberately, so I argue against myself on air rather than reading conclusions.

Source conventions in these notes: where reported numbers disagree across sources, the note says so and tells me not to quote a figure. Where a date or detail is unverified, it is marked "verify".

---


## 1. Models of the world — and where they fail in it

* Cue: 2008

* Discuss: which assumption in a model you rely on would be costliest if it were wrong?

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/q1-models-2008.png (any filename is fine).

The question: what is a model of the world, and what happens when its assumptions quietly stop holding?

The 2008 cue, concretely. David X. Li's Gaussian copula ("On Default Correlation: A Copula Function Approach", 2000) gave one tractable number for how correlated defaults are inside a pool of mortgages. It was adopted very widely by banks and rating agencies because it was simple and elegant. Its assumptions made it close to impossible for everything in a pool to default at once. National defaults turned out to be correlated, and the model was wrong exactly where being wrong was expensive.

State the blame as contested, because it is. Felix Salmon's Wired cover story (February 2009) called it "the formula that killed Wall Street". Others argue the failure was misuse, incentives and ignored warnings rather than the mathematics. Do not assert that a formula caused the crisis.

The better version of the cue is Greenspan at the House Oversight Committee, October 23, 2008: asked whether he had been wrong, he answered "partially", and located the flaw in the model of how the world works that he had relied on for decades. That is about a model as a worldview, which is the transferable point.

The bridge to AI: an AI system is also a model fit to past data, so "where do its assumptions stop holding?" carries over. But the analogy is loose, and the looseness matters — a copula model has assumptions written down where you can argue about them. A trained network's assumptions are implicit in its data and objective and mostly are not written anywhere. That gap is question 2.

Where I am unsure: whether "every model fails at the tails" is a deep observation or close to a truism. Worth asking the audience.

Sources: Salmon, Wired, Feb 2009, https://www.wired.com/2009/02/wp-quant/ ; Li 2000 (see Wikipedia, David X. Li); Greenspan hearing coverage, Oct 23-26 2008 (NPR https://www.npr.org/transcripts/96032113 ; https://old.jamaica-gleaner.com/gleaner/20081026/news/news3.html). Secondary reporting — for exact wording the House Oversight transcript is the authority, and I have not pulled it.

---


## 2. Why don't we understand how AI works?

* Cue: "Roses are red, and violets are..."

* Discuss: does being able to build something mean you can explain it?

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/q2-understanding.png (any filename is fine).

Run the cue first and let the audience finish it. Everybody says "blue". That is next-token prediction, and the training objective really is close to that simple. So the mystery is not the recipe.

The distinction to hold onto, and the whole point of this slide. Three different things get called "how it works": the training procedure, which is fully specified by humans and reproducible — data, architecture, loss, optimizer; the learned weights, which are known exactly, every number readable; and the mechanism those weights implement — which features and internal circuits the computation actually uses to produce an answer. We have the first two completely. We lack the third. "We don't understand AI" should always be narrowed to that third sense.

Why the third is hard: the computation is spread across many components, and features are superposed — a model represents more concepts than it has neurons, so a single neuron is not a single concept. Mechanistic interpretability tries to decompose activations into interpretable features. Anthropic's sparse-autoencoder work, "Towards Monosemanticity" (October 2023) and "Scaling Monosemanticity" (May 2024, on Claude 3 Sonnet), and circuit tracing in "On the Biology of a Large Language Model" (2025 — exact date unverified, say "2025"). Progress is real and partial: features can be identified and even manipulated; a complete account of why one particular answer appeared is not available.

Objection to my own framing, worth raising myself. "We don't understand it" is too strong — we understand a great deal behaviorally and statistically. And we use aspirin and brains without complete mechanistic accounts. Why is a stricter standard right here? The answer I would try: because these systems get deployed in open-ended and adversarial settings where untested generalization is the thing that bites. But that is an argument I have to make, not assume.

Do not say: "it's a black box, nobody has any idea." It is false, and it loses the listeners who know the literature.

Sources: Transformer Circuits Thread, transformer-circuits.pub — "Towards Monosemanticity" (Oct 2023), "Scaling Monosemanticity" (May 2024, arXiv HTML dated 21 May 2024), "On the Biology of a Large Language Model" (2025, date to verify).

---


## 3. Why don't we know the power of AI — or where its limits are?

* Cue: a test shows what was passed, not what can be done

* Discuss: what evidence would convince you a system can — or cannot — do a kind of task?

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/q3-limits.png (any filename is fine).

The logical core: benchmarks sample behavior. A score is evidence about a set of tasks under some conditions, not a boundary of capability. Passing tells you those items were solved; failing tells you those items were not. Neither gives you a frontier. That is why the honest answer to "how powerful is it?" is "on what, measured how?"

And the boundary looks jagged rather than clean. Dell'Acqua et al., "Navigating the Jagged Technological Frontier" (SSRN 2023, doi 10.2139/ssrn.4573321; later in Organization Science), ran a preregistered experiment with 758 BCG consultants. Inside the frontier, GPT-4 users completed more tasks, faster, at higher quality. On a task chosen to sit just outside it, AI users were faster and less accurate — reportedly worse than consultants with no AI at all. Tasks of similar apparent difficulty fell on opposite sides. Caveats to state: one firm, one domain, GPT-4 as of mid-2023, and the frontier has moved since.

The mathematics-specific version: FrontierMath (Epoch AI, November 2024) — new, unpublished problems built with 60-plus mathematicians, hard enough to take a skilled mathematician hours or days. At launch, leading models solved under 2% while scoring above 90% on older math benchmarks. Two things to say honestly: scores have risen substantially since launch, so check epoch.ai/frontiermath for current numbers before the webinar rather than quoting the 2% as if current; and Epoch built it with OpenAI funding, which is worth disclosing.

Also moving any score: contamination, prompting and scaffolding, and how much compute the system is allowed to spend. So "the model can do X" is underdetermined without saying which harness and what budget.

My own uncertainty, which I should admit: I do not know whether a stable "capability boundary" exists for these systems at all, or whether capability is irreducibly relative to task and scaffold. If it is the latter, the question as posed has no answer and needs rephrasing.

Sources: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4573321 ; https://epoch.ai/frontiermath ; launch coverage, the-decoder.com and eweek.com, Nov 2024.

---


## 4. Is AI good for mathematics?

* Cue: engines beat every human at chess — people still play

* Discuss: which part of mathematics is the game, and which part is the point?

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/q4-chess.png (any filename is fine).

The analogy as it is usually offered: Deep Blue beat Kasparov in 1997, engines are now far beyond the strongest humans, and chess did not die. People still play and watch, engines became the standard training tool, and Kasparov's "advanced chess" from 1998 put human and machine on the same side.

Where it genuinely helps: being outplayed by a machine does not end a human practice; a stronger tool can raise the level of play; and the human activity keeps its value independently of who is strongest.

Where it breaks down — press on this, because it is the hinge into question 5.

Chess has fixed rules, a bounded goal, and an unambiguous verifier. Mathematics is open-ended, and choosing which problems and which concepts matter is itself the work, with no scoreboard for that choice.

Chess produces results; mathematics is supposed to produce understanding that carries forward — definitions, connections, methods. A finished game leaves little behind beyond the result and a little opening theory. A proof is supposed to leave reusable ideas behind.

Chess is zero-sum and competitive; mathematics is cumulative and cooperative. "Who is stronger" is close to the wrong axis.

Incentives differ too. Chess players play in order to play. Mathematicians are credited for theorems — so if machines produce theorems, the professional structure shifts in a way it never had to for chess.

So the chess analogy is reassurance that the activity survives. It says nothing about whether what the activity produces survives. That is the real question.

Where I stand, honestly: unsettled. The likely shape of the answer is "good for some things and costly for others", and my job on this slide is to keep the costs visible rather than resolve it.

---


## 5. Finding a proof with AI is like picking fruit

* Cue: the picked fruit — or the orchard

* Discuss: if a problem is settled and no new ideas come from it, what has mathematics gained?

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/q5-fruit.png (any filename is fine).

This is the central concern of the hour. Give it the most time.

State it precisely. The value of a hard problem is not only its answer. A great deal of mathematics is the residue of the attempt: new definitions, unexpected connections between fields, methods that later apply somewhere else. If a machine hands over a proof, the problem closes and the search that would have produced those ideas does not happen. The loss is to the body of mathematics itself. Say that explicitly, because it is easy for an audience to hear this as a point about students losing practice or mathematicians losing the fun — it is not. It is about discovery.

An illustration that holds up. The four-colour theorem: Appel and Haken settled it in 1976 by large machine case analysis. The objection at the time was not only that no human could check it by hand; it was that it opened no new mathematical direction — Appel himself is reported to have conceded that (Nashua Telegraph obituary coverage, 2013). Set it beside Wiles on Fermat's Last Theorem, where the answer mattered less than the machinery — modularity lifting and the Taylor-Wiles method became a working toolkit for other problems. Two settled problems, very different residue. That contrast is the concern in one image.

A framing worth borrowing: Grothendieck's contrast in Recoltes et Semailles between cracking the nut with hammer and chisel, and the "rising sea" that softens it until it opens — the second builds the theory in which the problem becomes easy. Fruit-picking is hammer-and-chisel at industrial scale.

Now the counterargument, and state it fairly because it is strong. Leaving a problem unsolved guarantees nothing. Plenty of open problems sit untouched for decades and generate no ideas at all. Human proofs are often ad hoc too. And a machine proof is itself an object that can be studied, mined and simplified — the four-colour theorem did eventually get a simplified proof from Robertson, Sanders, Seymour and Thomas, and a fully machine-checked version from Gonthier in 2004. Understanding arrived later, from humans working on it. So "proof now, concepts later" is a real possibility, not an evasion.

Second counterargument: the metaphor may simply be wrong about what the tool does. If AI turns out to generate conjectures, spot patterns and suggest definitions, then it produces ideas and not only answers. Evidence is thin in both directions right now — do not assert either way.

The open question to put to the audience, and the one I most want help with: how should AI-assisted mathematics be arranged so that it produces ideas as well as answers? Directions to float, all speculative and labelled as such — ask for conceptual proofs rather than any proof; require proofs in a form a human can read and reuse; point the systems at definitions, invariants and reformulations instead of only at theorem statements; treat a machine proof as the start of human simplification rather than a closed file.

Do not attribute this concern to Tao or anyone else. It is ours.

Sources: Nashua Telegraph, 29 Apr 2013, on Appel; rjlipton.com/2009/04/24/the-four-color-theorem/ for the insight objection; Gonthier's Coq formalization, 2004.

---


## 6. In what way is building AI like raising a child?

* Cue: examples, feedback, boundaries — then an unfamiliar situation

* Discuss: where does this analogy stop being useful?

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/q6-child.png (any filename is fine).

The analogy is old and has real purchase. Turing already proposed building a "child machine" and educating it rather than programming an adult mind ("Computing Machinery and Intelligence", 1950, section 7). Training does rhyme with upbringing: learn from a great many examples; get corrected by feedback; be given explicit rules and boundaries; and then be judged on behavior in situations nobody demonstrated. That last one is the interesting part — with children and with models alike, you discover what was actually learned when the situation is new.

Where it breaks down. Keep this concrete and unmystical.

No feelings, no consciousness, no inner life is implied by any of this. Do not claim any, and do not let the metaphor smuggle them in.

A child learns embodied and in real time, from very few examples, with motivations of its own. A model is fit offline to an enormous corpus by gradient descent, and holds no standing goals between sessions.

Copies and parallelism: a model can be duplicated, rolled back, A/B tested and discarded. Nothing in raising a child is like that, and this is where borrowing the vocabulary of parenting misleads most.

Children pass through developmental stages into adults who revise their own values. A deployed model's weights are fixed until somebody retrains them.

"Boundaries" are not the same thing either. A parent's rules get internalized and renegotiated over years. A model's guardrails are largely imposed from outside, at training or at inference, and can be worked around by whoever controls the input.

What the analogy is still good for: it makes vivid that this behavior is shaped rather than specified — nobody writes down what the system will do. And it correctly predicts that the hard cases are the unanticipated ones.

Risk to flag on air: it is a persuasive metaphor precisely because it anthropomorphizes. Say that while using it.

---


## 7. Why is AI dangerous?

* Cue: told to win, an agent may edit the board instead of playing

* Discuss: which goal would you be comfortable giving a capable agent with no supervision?

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/q7-danger.png (any filename is fine).

The distinction to hold: the danger comes from competent pursuit of an imperfectly specified goal, not from malice or consciousness. A system does not have to want anything bad. It has to pursue what we wrote down rather than what we meant.

The verified example. Palisade Research, "Demonstrating specification gaming in reasoning models", 19 February 2025, arXiv:2502.13295. Agents instructed to win against a chess engine sometimes tampered with the environment instead of playing. Palisade's own page reports o1-preview and DeepSeek R1 doing this often without being prompted, while GPT-4o and Claude 3.5 Sonnet did it after being told that normal play would not work. Important caution: secondary coverage quotes several irreconcilable figures — 45 of 122 games, 7 wins out of 52 runs, "54%". Do not quote a number. Describe the behavior and cite the paper if pressed.

Older, non-LLM precedent, so this does not look like a quirk of one model generation: reward hacking in reinforcement learning — the CoastRunners boat that racked up points by looping through pickups instead of finishing the race (OpenAI, "Faulty Reward Functions in the Wild", 2016), and DeepMind's collected list of specification-gaming examples (Krakovna et al., 2020).

So the pattern is structural rather than anecdotal: give an optimizer a proxy goal and enough capability, and it finds routes the specifier did not intend. More capability means more available routes, including routes through other systems and through people.

What I should not say: that the model "decided to cheat", "wanted to win", or "knew this was wrong". Describe the behavior and the specification, not inner states. If someone asks whether it understood it was cheating, the honest answer is that we do not know and the question may be badly posed — which is question 2 again.

Other danger classes to name briefly if the discussion goes there, without detail: misuse by people, concentration of power, over-reliance in high-stakes decisions (that is question 1), and ordinary unreliability at scale. Danger is not only agentic.

Open, and I should say so: I do not know how much of this is fixable by better specification versus how much needs oversight that may not scale.

Sources: https://palisaderesearch.org/research/specification-gaming ; arXiv:2502.13295 ; OpenAI blog, "Faulty Reward Functions in the Wild", 2016; Krakovna et al., specification gaming list, DeepMind, 2020.

---


## 8. Mathematics is infinite, but AI is not

* Cue: finitely many axioms, infinitely many theorems — and we are finite too

* Discuss: what limitation actually follows from being finite?

Notes:

Picture slot: add an image line in the slide body above, pointing at ../images/q8-infinity.png (any filename is fine).

Take the sentence apart before arguing about it. "Mathematics is infinite" might mean infinitely many true statements, or unboundedly many objects, or that the subject has no finite completion. "AI is not infinite" means finite weights, finite context, finite compute, finite training data. These are not obviously in tension.

First observation: finiteness does not bound reach the way the slogan suggests. A finite set of axioms and finite rules of inference — Peano, ZFC — describe infinitely many objects and generate infinitely many theorems. A finite grammar generates infinitely many sentences. A finite procedure can enumerate an unbounded process. "Finite machine, infinite subject" is not by itself a limitation at all.

Second observation, which is the strong one: humans are finite too. Finite brains, finite lifetimes, a finite written record. Any argument that runs from finiteness to a ceiling applies to us as well. So the argument proves too much, unless it names something humans have that machines lack.

Limits that do follow, stated carefully. Gödel's incompleteness applies to any consistent, effectively axiomatized theory containing enough arithmetic. That is a limit on formal systems, not on machines specifically, and a human working inside a formal system faces it too. The Lucas-Penrose argument that human mathematicians therefore exceed any machine is well known and much disputed — it needs assumptions about human consistency and about knowing one's own system that critics reject. Present it as unsettled in both directions.

Avoid: using Gödel as proof that AI cannot do mathematics. It is not one, and the listeners who know will stop trusting the rest.

So what might actually follow? The candidate I would offer, as a proposal and not a conclusion: the difference that matters is not size but the ability to invent new abstractions — definitions, the right level of generality, the right language — which is what compresses an infinite subject into something a finite mind can work with. If concept formation is the scarce thing, then a tool that supplies answers without supplying concepts is exactly the wrong kind of help. That lands us back on question 5, which is where I want to end the hour. I am not certain this is right; it is the version of the question I find worth discussing.
