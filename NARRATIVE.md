# The Decision: A Walk-Through

This file tells the story of the decision Experiment Architect is built for. If
you want the code structure, read [ARCHITECTURE.md](ARCHITECTURE.md). If you
want the method-by-method notes, the README's method-choice and limitations
sections cover them. This document is neither: it is the *decision* itself, why
it was hard, what the tool said, and what you would actually do with the output
in a real review.

The scenario below is a representative composite, generalized from real
engagements and safe to share. The numbers are illustrative, chosen so you can
reproduce every step in the app's design tab. The point is the decision
pattern, not the figures.

---

## The situation

A team wants to ship a change to the checkout flow. Conversion on that surface
runs at about 4%. The change is a plausible improvement (a clearer step, a
removed field), the kind of thing that might lift conversion by a few percent
relative if it works at all. Leadership wants evidence before it rolls to
everyone, and there is a fixed decision window: three weeks, then the quarter's
planning locks.

The instinct in the room is the default one: run an A/B test, wait for
significance, ship if the p-value clears 0.05. Nobody in the room thinks that is
controversial. It is the wrong move here, and the reason is arithmetic, not
philosophy.

---

## Why "run it and read the p-value" is the wrong move here

Two bad options usually sit on the table when a test is tight, and both get
taken:

- **Run it anyway and peek early.** The test starts, the dashboard updates
  daily, and someone reads a promising day-four result as a green light. Peeking
  at an underpowered test is the classic way to ship on noise. The result is
  directional at best and an inflated false-positive at worst.
- **Skip the rigor and go with judgment.** "We can't really test this, so let's
  just decide." That throws away the evidence the test *could* have produced and
  hides the risk instead of bounding it.

Neither is acceptable, and picking between them is the wrong frame. The first
question is not "frequentist or Bayesian" or "ship or hold." It is: *can this
test detect the effect we actually expect, in the time we actually have?* That
is the question the design tab answers first, before any data exists.

---

## What the reverse-MDE check showed

Sample-size math usually runs forward: pick an effect you want to detect, get
the sample size you need. Reverse MDE runs it backward: take the sample you will
realistically have, and report the smallest effect it could detect. That is the
honest direction when traffic is fixed and the deadline is not moving.

The eligible traffic to the checkout surface is about 12,000 users a week, split
evenly, so roughly 6,000 per arm per week. Over the three-week window that is
about 18,000 per arm. Feed that into the design tab at a 4% baseline, 80% power,
and a two-sided 5% test, and the reverse-MDE read-out is blunt:

> The smallest effect this test could detect is about a **15% relative lift**
> (roughly 0.6 points of conversion, from 4.0% to 4.6%).

But the change is expected to move conversion by a few percent relative, not
fifteen. The test is powered to catch an effect several times larger than the
one anyone believes is there. Ask the tool the forward question to confirm it:
to power a 3% relative lift at this baseline, each arm needs on the order of
400,000 users. At 6,000 per arm per week, that is well over a year. The
three-week A/B was never going to answer the question. It only looked like it
would.

This is the moment the tool earns its keep. It does not produce a result. It
tells you, before you spend three weeks and the credibility of a launch on it,
that the experiment in front of you cannot answer the question you are asking.

---

## The reframe: answer a question the test can answer, then decide on loss

Once the primary A/B is off the table, the decision does not collapse into "go
with your gut." It splits into two questions the tool *can* handle.

**A guardrail question the test can answer.** Detecting a small lift needs
enormous power. Detecting a large *regression* does not. Scoped to "does this
change clearly hurt conversion or revenue," the three weeks of data is enough:
the same design tab shows a 1-point regression is well within detectable range.
So the test is re-pointed from "does it help?" (unanswerable here) to "does it
clearly hurt?" (answerable). A pre-registered guardrail on the revenue metric
comes along for free.

**A ship decision framed as expected loss, not significance.** With the
guardrails clear, the ship call moves to the Bayesian read-out. Instead of a
binary "significant / not significant," the app reports two things a decision
maker can act on: the posterior probability that the variant is at least as good
as control, and the expected loss if you ship it and you turn out to be wrong.
In this scenario the posterior sits around 0.7 (encouraging, nowhere near
frequentist significance) and the expected loss of shipping is a couple of basis
points of conversion, because the guardrails have already ruled out a large
regression. The expected loss of *not* shipping a cheap, plausibly-positive
change is larger. The rule the team agreed to in advance ("ship if expected loss
stays under this threshold and no guardrail trips") returns: ship, and watch the
guardrails.

The important boundary, the same one enforced in the code, is that the tool
never hides the uncertainty. It states the win probability *and* the loss, so
the team ships knowing exactly what it is and is not sure about.

---

## What changed

The decision stopped depending on a test that could not deliver in time. Instead
of a doomed A/B read through peeking, or a rigor-free judgment call, the team
shipped on an explicit, loss-aware rule agreed before anyone saw a number, with
guardrails watched rather than ignored. The conversation in the review moved off
"is it significant yet?" and onto "what is our tolerance for being wrong, and
have the guardrails held?" That is a conversation a leadership team can actually
close.

---

## What this tool can and cannot tell you

- **It sizes the question honestly; it does not invent power you do not have.**
  Reverse MDE tells you the test is underpowered. It cannot make a three-week
  window into a year of traffic. The answer is sometimes "this cannot be an A/B,"
  and the tool's job is to say so early.
- **The Bayesian read-out needs an honest prior and an honest loss.** The ship
  rule is only as good as the loss threshold agreed up front. Set that in
  planning, not at read-out, or the number gets reverse-engineered to match the
  decision someone already wants.
- **Guardrail diagnostics are warnings, not proofs.** When randomization is not
  possible at all and the app routes to DiD or RDD, the parallel-trends and
  density checks flag problems; passing them does not prove identification. The
  tool surfaces the diagnostic so you argue with it, rather than burying it.
- **The LLM never computes the statistic.** It maps messy column names to roles
  so Python can run the maths. A human still owns whether the mapping is
  semantically right.

---

## How to use this pattern in a real review

Three practical suggestions:

**Run reverse MDE before you commit the traffic, not after.** The single most
useful thing this approach produces is the early "this test cannot detect the
effect we expect" finding. It is cheap before launch and expensive after.

**Agree the acceptable cost of being wrong in planning.** The loss-aware ship
rule lands as a finding rather than a surprise only if the loss tolerance was set
before anyone had a result they liked. Anchor the risk appetite first, then let
the read-out drive the call.

**When you can't test, choose the fallback deliberately.** "We can't A/B this"
is the start of the analysis, not the end. Sometimes the answer is a guardrail
question the test *can* answer, sometimes a loss-aware ship rule, sometimes a
causal method with its diagnostics on show. The tool exists to make that choice
explicit instead of defaulting to peeking or to gut feel.

---

Experiment Architect is one of three headline projects in the
[Product Decision Lab](https://github.com/tomasz-solis/product-decision-lab).
The companion decision-analysis case, [Product Decision Under
Uncertainty](https://github.com/tomasz-solis/product-decision-under-uncertainty),
walks through the same discipline applied to an irreversible platform
investment.
