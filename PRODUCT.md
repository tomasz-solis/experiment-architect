# Product

## Register

product

## Platform

web

## Users

Product leaders facing one real, irreversible call — the same person the rest of the Product
Decision Lab is built for. Here they are one step upstream: they intend to settle the question
with an experiment, and they need to know whether the experiment can actually answer it before
they spend the traffic. They arrive with a baseline, a rough idea of the lift they care about,
and a deadline. They are not statisticians.

There is one audience. If it works for the leader with a live decision, it works.

## Product Purpose

A pre-flight tool for experiment design: it tells you whether the test you are about to run can
detect the effect you care about with the traffic you actually have — and what to do when it
cannot. It covers power and sample size, a reverse-MDE audit, frequentist guardrails, sensitivity
to traffic and baseline assumptions, Bayesian posteriors, and CSV-driven analysis of results that
already exist.

Success is that the tool is useful enough on someone's real experiment that they get in touch.
The most valuable outcome it can produce is often a test that never runs.

## Positioning

Know whether an experiment can answer your question **before** you spend the traffic on it — and
what to do when it can't. The second half of that sentence is the part nothing else does.

## Brand Personality

Rigorous, welcoming, practical. The voice is an honest analyst briefing a decision-maker, not a
vendor pitching one: it states the method, names its own limits out loud, and assumes the reader
is smart but not a statistician. Closest in feel to FT / Economist interactives — editorial
numeracy, where the chart and the prose argue together and the honest caveat is part of the
argument rather than a footnote.

## Anti-references

A raw Streamlit default: unstyled widgets, stock primary colours, the look of a prototype nobody
cared about. If the interface reads as "someone's weekend notebook with a slider on it", the
method's credibility goes with it.

## Design Principles

- **A test that shouldn't run is a result.** The tool's best answer is often "this cannot work" —
  design for that outcome as a first-class success, never as an error state.
- **Say the limit out loud.** Where the method is weak or the assumptions are thin, the interface
  says so at the point of reading. Honesty is the trust mechanism.
- **Meet a non-statistician where they stand.** MDE, power, and posterior all carry a
  plain-language definition within reach. Jargon that cannot explain itself does not ship.
- **Their numbers are the proof.** The bring-your-own-data path earns the decision; keep it a
  first-class route, not an expert escape hatch.
- **One system across the Lab.** This app is one of three Product Decision Lab surfaces
  (`product-decision-under-uncertainty`, `experiment-architect`,
  `measurement-maturity-framework`). They must read as one family, so the design system is a
  shared contract, not a local choice — see the constraint below. A visual idea that cannot
  travel to the other two does not ship here.

## Design system constraint (cross-repo)

`ui/theme-tokens.css` is the shared source of truth and is **byte-identical** in all three repos
(stored at `ui/` here, `static/` in product-decision-under-uncertainty, `mmf/` in
measurement-maturity-framework). Also shared: the pinned light Streamlit base with
`primaryColor #4f6dff`, the periwinkle/mint radial app background, the dark gradient sidebar,
glassy light panels, pill tabs and buttons, and the Avenir Next stack.

Rules:

- Never edit the `--ds-*` tokens for this app alone. A token change is a three-repo change:
  edit it here, then copy the file verbatim into the other two.
- Local names (`--blue`, `--mint-text`, …) may only alias `--ds-*` tokens, never redefine the palette.
- Per-app variation is allowed only in the branded hero (`.editorial-hero`, vs `.app-hero` and
  `.mmf-hero`) and in app-specific components. Everything else stays in the family vocabulary.

## Accessibility & Inclusion

WCAG 2.1 AA. Body text ≥4.5:1 and large text ≥3:1 against its surface (the dark sidebar and the
dark hero included), full keyboard navigation, `prefers-reduced-motion` honoured, and charts that
stay readable when colour is unreliable — every series pairs its colour with a dash or marker.
