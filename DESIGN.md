---
name: Experiment Architect
description: Know whether an experiment can answer your question before you spend the traffic on it — one of three Product Decision Lab surfaces sharing a single design system.
colors:
  periwinkle: "#4f6dff"
  periwinkle-strong: "#3a56e0"
  periwinkle-soft: "#4f6dff1f"
  mint: "#1ecf9b"
  mint-deep: "#0a7d5c"
  amber: "#d18a1f"
  amber-deep: "#9a6a12"
  red: "#dd5b52"
  red-deep: "#c43d33"
  ink: "#10131a"
  muted: "#646c79"
  bg: "#eff2f7"
  bg-deep: "#e7ebf3"
  surface: "#ffffffcc"
  line: "#10131a14"
  console-top: "#0d1320"
  console-bottom: "#101925"
  console-ink: "#f8fbff"
typography:
  display:
    fontFamily: "Avenir Next, Segoe UI, Helvetica Neue, system-ui, -apple-system, sans-serif"
    fontSize: "clamp(2.1rem, 3.6vw, 3.1rem)"
    fontWeight: 800
    lineHeight: 1.0
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "Avenir Next, Segoe UI, Helvetica Neue, system-ui, -apple-system, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.03em"
  title:
    fontFamily: "Avenir Next, Segoe UI, Helvetica Neue, system-ui, -apple-system, sans-serif"
    fontSize: "1.45rem"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Avenir Next, Segoe UI, Helvetica Neue, system-ui, -apple-system, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "normal"
  label:
    fontFamily: "Avenir Next, Segoe UI, Helvetica Neue, system-ui, -apple-system, sans-serif"
    fontSize: "0.82rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "0.01em"
rounded:
  control: "18px"
  card: "24px"
  hero: "28px"
  pill: "999px"
spacing:
  gap: "0.85rem"
  panel: "1rem"
  section: "1.6rem"
components:
  button-primary:
    backgroundColor: "{colors.periwinkle}"
    textColor: "#ffffff"
    rounded: "{rounded.pill}"
    padding: "0.7rem 1.15rem"
    typography: "{typography.label}"
  button-primary-hover:
    backgroundColor: "{colors.periwinkle-strong}"
    textColor: "#ffffff"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "0.7rem 1.15rem"
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.card}"
    padding: "1rem 1.05rem"
  hero:
    backgroundColor: "{colors.console-top}"
    textColor: "{colors.console-ink}"
    rounded: "{rounded.hero}"
    padding: "1.9rem 2.1rem"
  tab-selected:
    backgroundColor: "{colors.ink}"
    textColor: "#ffffff"
    rounded: "{rounded.pill}"
    padding: "0.45rem 0.95rem"
  tab-idle:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "0.45rem 0.95rem"
  input:
    backgroundColor: "#ffffff1f"
    textColor: "{colors.console-ink}"
    rounded: "{rounded.control}"
    padding: "0.5rem 0.75rem"
---

# Design System: Experiment Architect

> **Shared system.** Sections 1–5 of this file are the Product Decision Lab design system and are
> **identical in all three repos**: `product-decision-under-uncertainty`, `experiment-architect`,
> `measurement-maturity-framework`. The token file behind them (`--ds-*`) is byte-identical in all
> three, stored at `static/theme-tokens.css` here, `ui/theme-tokens.css` in experiment-architect, and
> `mmf/theme-tokens.css` in measurement-maturity-framework. Change a token here, copy it there. A
> visual idea that cannot travel to the other two does not ship.

## 1. Overview

**Creative North Star: "The Briefing Room"**

A dark banner frames the question; light panels lay the evidence on the table. That is the whole
system in one sentence. The user arrives with a decision they have to defend to someone else, so the
interface is staged like a briefing: the hero states what is being decided, and everything below it
is material you could put in front of a room without apologising for it. The dark console — hero and
sidebar — is where the app speaks. The light field below is where the evidence lives. Nothing else
in the system is allowed to compete with that division.

The register is a tool, not a brochure. Density is welcome; decoration is not. The interface should
feel like an instrument that happens to be well made, not a product that is selling you something.
Type is one family across every surface, colour is used as signal, and depth is spent only where it
carries meaning. When in doubt, the honest, plainer option wins — this system's credibility is the
product's credibility, and a chart that flatters is worse than no chart.

This system explicitly rejects **the raw Streamlit default**: stock widgets, the stock red primary,
the untended look of a prototype nobody cared about. It equally rejects the opposite failure — the
SaaS dashboard that hides thin reasoning behind gradients and vanity metrics. Neither reads as
serious to someone with a real decision on the table.

**Key Characteristics:**
- Dark console (hero + sidebar), light evidence field. One division, held everywhere.
- One type family (Avenir Next), one fixed rem scale. No display face, no pairing.
- Periwinkle is the Lab's signature; mint is reserved and means *passed*.
- Flat at rest. Depth is a response to state, not a default texture.
- Never encode meaning in hue alone.

## 2. Colors: The Instrument Palette

Periwinkle and mint on a cool grey field: the palette of a lit instrument, not a brand deck. Cool
neutrals carry the surface; the two brand hues carry identity and state.

### Primary
- **Instrument Periwinkle** (`#4f6dff`): the Lab's signature. Primary actions, current selection,
  focus, the first chart series, and the wash inside the hero. As the family identity marker it is
  permitted more surface than a typical product accent — it may carry heroes, headers, and a chart
  series — but it never becomes a decorative texture applied to things that are not interactive or
  not the subject.
- **Deep Periwinkle** (`#3a56e0`): the AA-safe text form. Any periwinkle *text* on a light surface
  uses this, never `#4f6dff`, which fails 4.5:1 as body copy.

### Secondary
- **Clearance Mint** (`#1ecf9b`): reserved. Mint means a threshold was cleared, a check passed, a
  state is healthy. It is never a second decorative accent, never a gradient partner, never "the
  other brand colour". Its meaning is the point.
- **Deep Mint** (`#0a7d5c`): the AA-safe text and stroke form of mint on light surfaces.

### Tertiary
- **Caution Amber** (`#d18a1f`) / **Deep Amber** (`#9a6a12`): a warning, a caveat, a stale
  fingerprint, a governance banner. Never a highlight.
- **Risk Red** (`#dd5b52`) / **Deep Red** (`#c43d33`): a breach, a failure, a guardrail not met.
  Never used for emphasis or for "important".

### Neutral
- **Ink** (`#10131a`): all primary text on light surfaces; also the fill of a selected tab.
- **Muted Slate** (`#646c79`): secondary text, captions, baseline chart series. This is the floor —
  do not go lighter for body text. Lighter greys fail contrast and read as carelessness.
- **Field** (`#eff2f7`) and **Deep Field** (`#e7ebf3`): the app background, a cool grey gradient.
- **Panel** (`#ffffffcc`): the evidence surface — white at 80%, over the field.
- **Hairline** (`#10131a14`): every panel border. 1px. Always.
- **Console** (`#0d1320` → `#101925`): the dark gradient of the sidebar and hero.
- **Console Ink** (`#f8fbff`): text on the console.

### Chart series (shared, all three apps)
One categorical ramp, in this order, chosen so adjacent series differ in lightness as well as hue:
**Instrument Periwinkle** (`#4f6dff`) → **Deep Mint** (`#0a7d5c`) → **Deep Amber** (`#b5741a`) →
**Deep Red** (`#c43d33`) → **Muted Slate** (`#646c79`, reserved for the baseline / do-nothing
series). Gridlines `#dde3ec`, axis and tick text Ink, hover surface white.

### Named Rules
**The Signature Rule.** Periwinkle is identity and may carry surface. Mint is meaning and may not.
The moment mint appears on something that did not pass a check, the system has started lying.

**The Redundant Encoding Rule.** No meaning may be carried by hue alone — not in a chart, not in a
status pill, not in a guardrail verdict. Colour always rides alongside a label, a shape, a dash
pattern, or an icon. This is a WCAG AA requirement and a decision-integrity requirement at once: a
reader who cannot see the green must still be able to see that it passed.

**The Deep-Form Rule.** Every brand hue has a light form (fills, marks, backgrounds) and a deep form
(text, strokes). Text on a light surface always takes the deep form. `#4f6dff` as body copy is a bug.

## 3. Typography

**Display Font:** none. This system has no display face.
**Body Font:** Avenir Next (falling back to Segoe UI, Helvetica Neue, system-ui, -apple-system, sans-serif)
**Label/Mono Font:** none distinct; code and figures inherit the same stack, with the browser mono
default reserved for literal code strings.

**Character:** One humanist sans doing every job, from the hero headline down to the axis tick. The
personality comes from weight and negative tracking on the large sizes, not from a second face. A
serif or display pairing here would read as a brochure; the tool would lose the room.

### Hierarchy
- **Display** (800, `clamp(2.1rem, 3.6vw, 3.1rem)`, 1.0, `-0.04em`): the hero title only. The single
  place fluid type is permitted, because the hero is a banner, not UI.
- **Headline** (700, 1.75rem, 1.2, `-0.03em`): a page-level heading below the hero.
- **Title** (700, 1.45rem, 1.3, `-0.02em`): section headings — "Guardrail eligibility", "Policy frontier".
- **Body** (400, 1rem, 1.6): explanatory prose. Capped at 62rem (~70ch); prose never runs the full
  1440px container.
- **Label** (600, 0.82rem, 1.4): control labels, table headers, chart legends. Sentence case.

### Named Rules
**The One Family Rule.** Avenir Next carries headings, buttons, labels, body, and data. Adding a
second family is prohibited. Contrast comes from weight (400 / 600 / 700 / 800) and size.

**The Fixed-Scale Rule.** Every size in the UI is a fixed rem step. `clamp()` is permitted in exactly
one place — the hero title. A section heading that resizes with the viewport is a brand-page habit
and does not belong in a tool.

**The Eyebrow Rule.** Tiny uppercase letter-spaced labels (`0.74rem`, `0.12em`+, uppercase) are a
scaffold, not a voice. At most **one** per screen, in the hero, and only when it names something real.
Stacking an uppercase eyebrow above every card, note, and metric is the tell of a generated layout;
sentence-case labels at 0.82rem do the same job without the costume.

## 4. Elevation

Flat at rest. The evidence field is a plane of hairline-bordered panels sitting directly on the
background — no shadow, no float, no glass. Depth in this system is **a response to state**, never a
default texture: it appears when an element is hovered, when it overlays other content, or when it is
the console speaking. Everything else earns its separation from the background with a 1px hairline
(`#10131a14`) and the tonal step between panel white and the cool grey field.

The failure mode this replaces is "everything is a floating card": when every panel carries a
48–72px shadow, nothing reads as raised, the page acquires a soft blur of drop shadows, and depth
stops carrying information.

### Shadow Vocabulary
- **Console** (`box-shadow: 0 30px 72px rgba(12, 16, 24, 0.22)`): the hero banner only. It is the one
  element allowed to sit visibly above the page.
- **Overlay** (`box-shadow: 0 16px 36px rgba(15, 23, 42, 0.14)`): things that genuinely float over
  content — dropdowns, popovers, modals, the file-uploader menu.
- **Hover lift** (`box-shadow: 0 12px 24px rgba(79, 109, 255, 0.12)`, with `translateY(-1px)`):
  interactive elements, on hover only. Removed under `prefers-reduced-motion`.
- **Rest**: none. A `border: 1px solid` hairline instead.

### Named Rules
**The Flat-At-Rest Rule.** A panel, a chart, a table, a metric card, and an expander all sit flat.
If a surface has a shadow at rest, delete the shadow.

**The No-Glass Rule.** `backdrop-filter: blur()` is prohibited as decoration. Translucency is a
material for overlays only, and even there it must be justified by something real sitting behind it.

**The Audit Test.** Squint at the page. If you see a field of soft grey halos rather than a flat
sheet of evidence, the shadows are back and they are wrong.

## 5. Components

Familiar affordances, immaculately kept. Nothing in this system reinvents a control; the tool should
disappear into the task. Every interactive component ships all of: default, hover, focus-visible,
active, disabled.

### Buttons
- **Shape:** fully rounded pill (`999px`).
- **Primary:** solid Instrument Periwinkle (`#4f6dff`) with white label, `0.7rem 1.15rem` padding,
  weight 600. Used for the one action that moves the decision forward on a given screen.
- **Secondary:** panel white with an ink label and a hairline border. Everything else — downloads,
  resets, template fetches.
- **Hover / Focus:** hover lifts 1px and takes the hover-lift shadow; `:focus-visible` shows a 2px
  Instrument Periwinkle ring at 2px offset. The focus ring is never removed, on any surface.
- **On the console:** a button sitting on the dark sidebar inverts to a solid white chip with ink
  text. Translucent buttons on the dark gradient are prohibited — they were the source of the
  invisible-label bug this system already fixed once.

### Cards / Containers
- **Corner Style:** 24px (`{rounded.card}`).
- **Background:** Panel white (`#ffffffcc`) over the field.
- **Border:** 1px Hairline. Always present — it is what does the work now that shadows are gone.
- **Shadow Strategy:** none at rest. See Elevation.
- **Internal Padding:** `1rem 1.05rem`.
- **Prohibited:** nested cards; a coloured accent stripe on any edge; a gradient bar across the top.

### Inputs / Fields
- **Style:** 18px radius, 1px border. On the light field: white fill, hairline border. On the console:
  white at 12% fill, white at 26% border, Console Ink text — and a real caret and placeholder colour,
  never the inherited dark default.
- **Focus:** border shifts to Instrument Periwinkle and the focus ring appears. No glow.
- **Error:** border and helper text take Deep Red; the helper text says what to do, not just what broke.
- **Disabled:** 55% opacity, no hover response.

### Navigation
- **Tabs:** pill-shaped (`999px`), sentence case, weight 600. Idle: panel white with a hairline border
  and ink label. Selected: solid **Ink** fill with a white label — the selection is a hard, unambiguous
  switch, not a tint. Hover on an idle tab: hairline darkens, no fill change. No blur, no glass.
- **Sidebar:** the console. Dark gradient (`#0d1320` → `#101925`), Console Ink labels, a hairline
  right border. It holds run settings and data source: the levers, never the evidence.

### The Console Hero (signature component)
The dark banner at the top of every Lab app — `.editorial-hero` here, `.app-hero` in
product-decision-under-uncertainty, `.mmf-hero` in measurement-maturity-framework. It is the one
place the system raises its voice.
- 28px radius, `1.9rem 2.1rem` padding, console gradient with a periwinkle radial wash top-left and a
  restrained mint wash top-right, Console shadow, 1px white-at-8% border.
- Contents, in order: at most one kicker (sentence case or small caps, never both uppercase *and*
  wide-tracked on every screen), the Display title, and one subtitle line capped at 48rem.
- The three apps' heroes differ only in copy and in the exact radial placement. Their structure,
  radius, padding scale, and shadow are the same. This is what makes the family legible as a family.

### Charts
- Series colours come from the shared categorical ramp (Section 2), in order, and are **paired with a
  non-colour channel**: distinct markers, dash patterns, or direct labels.
- Gridlines `#dde3ec`, hairline weight. Axis titles and ticks in Ink, no chart title where a section
  heading already says it.
- Chart surfaces obey Flat-At-Rest: the plot sits in a hairline panel, not a floating card.
- A chart never uses mint for a series that is not "passed"; see The Signature Rule.

## 6. Do's and Don'ts

### Do:
- **Do** keep `theme-tokens.css` byte-identical across the three repos. Edit it here, copy it verbatim
  into `experiment-architect/ui/` and `measurement-maturity-framework/mmf/`. A token is a three-repo change.
- **Do** alias the shared tokens locally (`--app-*`, `--mmf-*`, `--blue`) and never redefine their values.
- **Do** use Deep Periwinkle (`#3a56e0`), Deep Mint (`#0a7d5c`), Deep Amber (`#9a6a12`), Deep Red
  (`#c43d33`) for any coloured *text* on a light surface. The light forms are for fills and marks.
- **Do** pair every colour-coded meaning with a label, shape, or dash — guardrail pass/fail especially.
- **Do** give every panel a 1px hairline (`#10131a14`) and no shadow at rest.
- **Do** honour `prefers-reduced-motion`: the panel entrance animation and the hover lift both collapse
  to an instant state change. Transitions stay at 140–250ms; users are in a task, not watching a show.
- **Do** keep prose to ~70ch (62rem) even though the container runs to 1440px.
- **Do** invert controls to solid white chips when they sit on the dark console.

### Don't:
- **Don't** ship anything that reads as **a raw Streamlit default** — the stock red primary, unstyled
  widgets, an untended prototype. That is this product's stated anti-reference, and it takes the
  method's credibility down with it.
- **Don't** put a shadow on a resting surface. `0 18px 48px` under every chart, table, metric, and
  expander is the "everything floats" failure this system exists to correct.
- **Don't** use `backdrop-filter: blur()` decoratively — not on tabs, not on cards. Glass is not a texture.
- **Don't** put a gradient accent bar across the top of a metric card, or a coloured stripe down any
  edge of any surface. Both are decoration pretending to be information.
- **Don't** stack a tiny uppercase letter-spaced eyebrow above every section, note, and metric. One
  per screen, in the hero, or none.
- **Don't** use gradient text (`background-clip: text`) anywhere, ever.
- **Don't** use mint for anything that has not passed a check.
- **Don't** introduce a second font family, or a `clamp()` heading anywhere but the hero title.
- **Don't** let chart series drift off the shared ramp. Green/orange/slate palettes invented per-app
  are how three apps stop looking like one product.
- **Don't** encode a guardrail verdict, a scenario, or a series in colour alone.
