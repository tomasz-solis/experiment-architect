---
name: Experiment Architect
description: Know whether an experiment can answer your question before you spend the traffic on it. One of three Product Decision Lab apps sharing one design system.
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

# Design system: Experiment Architect

> Shared system. Sections 1 to 5 are the Product Decision Lab design system and are identical in `product-decision-under-uncertainty`, `experiment-architect` and `measurement-maturity-framework`. The token file behind them (`--ds-*`) is byte-identical in all three: `static/theme-tokens.css` in product-decision-under-uncertainty, `ui/theme-tokens.css` in experiment-architect, `mmf/theme-tokens.css` in measurement-maturity-framework. Change a token in one, copy it to the other two. A visual idea that can't work in all three doesn't ship.

## 1. Overview

North star: "The Briefing Room".

A dark banner states the question and light panels lay out the evidence. The user has a decision to defend to someone else, so the page works like a briefing: the hero says what is being decided, and everything below it could go in front of a room. The dark console (hero and sidebar) is where the app speaks. The light field below is where the evidence lives. Nothing competes with that split.

This is a tool, not a brochure. Density is fine, decoration isn't. Use one type family everywhere, use colour as a signal, and add depth only where it means something. When unsure, the plainer option wins. A chart that flatters is worse than no chart.

The system rejects two failures. One is the raw Streamlit default: stock widgets, the stock red primary, a prototype nobody looked after. The other is the SaaS dashboard that hides thin reasoning behind gradients and vanity metrics.

In short:

- Dark console (hero and sidebar), light evidence field, everywhere.
- One type family (Avenir Next) on one fixed rem scale. No display face.
- Periwinkle is the Lab's signature. Mint is reserved and means passed.
- Flat at rest. Depth responds to state.
- Never carry meaning in hue alone.

## 2. Colours: the instrument palette

Periwinkle and mint on a cool grey field. Cool neutrals carry the surface; the two brand hues carry identity and state.

| Role | Name | Value | Use |
|---|---|---|---|
| Primary | Instrument Periwinkle | `#4f6dff` | The Lab's signature: primary actions, current selection, focus, the first chart series, the hero wash. It may cover more surface than a normal accent (heroes, headers, a chart series), but never as decoration on things that aren't interactive or the subject. |
| Primary text | Deep Periwinkle | `#3a56e0` | Any periwinkle text on a light surface. `#4f6dff` fails 4.5:1 as body copy. |
| Secondary | Clearance Mint | `#1ecf9b` | Reserved: a threshold cleared, a check passed, a healthy state. Never decoration, never a gradient partner. |
| Secondary text | Deep Mint | `#0a7d5c` | Mint text and strokes on light surfaces. |
| Warning | Caution Amber / Deep Amber | `#d18a1f` / `#9a6a12` | Warnings, caveats, a stale fingerprint, a governance banner. Never a highlight. |
| Risk | Risk Red / Deep Red | `#dd5b52` / `#c43d33` | A breach, a failure, a guardrail not met. Never for emphasis. |
| Text | Ink | `#10131a` | Primary text on light surfaces, and the fill of a selected tab. |
| Secondary text | Muted Slate | `#646c79` | Captions, secondary text, the baseline chart series. Never go lighter for body text; lighter greys fail contrast. |
| Background | Field / Deep Field | `#eff2f7` / `#e7ebf3` | The app background, a cool grey gradient. |
| Surface | Panel | `#ffffffcc` | The evidence surface: white at 80% over the field. |
| Border | Hairline | `#10131a14` | Every panel border. Always 1px. |
| Console | Console | `#0d1320` to `#101925` | The dark gradient of the sidebar and hero. |
| Console text | Console Ink | `#f8fbff` | Text on the console. |

Chart series, shared by all three apps, in this order (adjacent series differ in lightness as well as hue): Instrument Periwinkle `#4f6dff`, Deep Mint `#0a7d5c`, Deep Amber `#b5741a`, Deep Red `#c43d33`, then Muted Slate `#646c79` for the baseline or do-nothing series. Gridlines `#dde3ec`, axis and tick text in Ink, white hover surface.

Rules:

- Signature rule: periwinkle is identity and may cover surface. Mint is meaning and may not. Mint on something that didn't pass a check is a lie.
- Redundant encoding rule: no meaning rides on hue alone, in a chart, a status pill or a guardrail verdict. Colour always comes with a label, shape, dash pattern or icon. This is a WCAG AA requirement and keeps decisions honest: a reader who can't see the green must still see that it passed.
- Deep-form rule: every brand hue has a light form (fills, marks, backgrounds) and a deep form (text, strokes). Text on a light surface always uses the deep form. `#4f6dff` as body copy is a bug.

## 3. Typography

- Display font: none.
- Body font: Avenir Next, falling back to Segoe UI, Helvetica Neue, system-ui, -apple-system, sans-serif.
- Label and mono font: none separate. Code and figures use the same stack; the browser mono default is only for literal code strings.

One humanist sans does every job, from the hero title to the axis ticks. Personality comes from weight and tighter tracking at large sizes. A serif or display pairing would look like a brochure.

| Level | Weight | Size | Line height | Tracking | Use |
|---|---|---|---|---|---|
| Display | 800 | `clamp(2.1rem, 3.6vw, 3.1rem)` | 1.0 | `-0.04em` | Hero title only. The one place fluid type is allowed. |
| Headline | 700 | 1.75rem | 1.2 | `-0.03em` | Page heading below the hero |
| Title | 700 | 1.45rem | 1.3 | `-0.02em` | Section headings ("Guardrail eligibility", "Policy frontier") |
| Body | 400 | 1rem | 1.6 | | Prose, capped at 62rem (about 70 characters), never the full 1440px container |
| Label | 600 | 0.82rem | 1.4 | | Control labels, table headers, chart legends, sentence case |

Rules:

- One family rule: Avenir Next carries headings, buttons, labels, body and data. No second family. Contrast comes from weight (400, 600, 700, 800) and size.
- Fixed scale rule: every UI size is a fixed rem step. `clamp()` is allowed only on the hero title.
- Eyebrow rule: tiny uppercase letter-spaced labels (`0.74rem`, `0.12em` or more) are allowed at most once per screen, in the hero, and only when they name something real. An uppercase eyebrow over every card and metric is the mark of a generated layout. Sentence-case labels at 0.82rem do the same job.

## 4. Elevation

Flat at rest. The evidence field is a plane of hairline-bordered panels on the background, with no shadow, float or glass. Depth appears only when an element is hovered, overlays other content, or is the console. Everything else is separated by a 1px hairline (`#10131a14`) and the tonal step between panel white and the grey field.

This replaces the "everything is a floating card" look. When every panel has a 48 to 72px shadow, nothing reads as raised and depth stops carrying information.

| Shadow | Value | Use |
|---|---|---|
| Console | `0 30px 72px rgba(12, 16, 24, 0.22)` | The hero banner only |
| Overlay | `0 16px 36px rgba(15, 23, 42, 0.14)` | Things that float over content: dropdowns, popovers, modals, the file uploader menu |
| Hover lift | `0 12px 24px rgba(79, 109, 255, 0.12)` with `translateY(-1px)` | Interactive elements on hover only. Removed under `prefers-reduced-motion`. |
| Rest | none | A 1px solid hairline border instead |

Rules:

- Flat at rest rule: panels, charts, tables, metric cards and expanders sit flat. A shadow at rest gets deleted.
- No glass rule: no decorative `backdrop-filter: blur()`. Translucency is only for overlays, and only when something real sits behind them.
- Squint test: if the page looks like a field of soft grey halos and not a flat sheet of evidence, the shadows are back.

## 5. Components

Familiar controls, well kept. Nothing reinvents a control. Every interactive component has default, hover, focus-visible, active and disabled states.

### Buttons

- Shape: full pill (`999px`).
- Primary: solid Instrument Periwinkle (`#4f6dff`), white label, `0.7rem 1.15rem` padding, weight 600. For the one action that moves the decision forward on a screen.
- Secondary: panel white, ink label, hairline border. For everything else: downloads, resets, templates.
- Hover and focus: hover lifts 1px with the hover-lift shadow. `:focus-visible` shows a 2px Instrument Periwinkle ring at 2px offset. The focus ring is never removed.
- On the console: a button on the dark sidebar becomes a solid white chip with ink text. Translucent buttons on the dark gradient caused an invisible-label bug once, so they are banned.

### Cards and containers

- Corners: 24px (`{rounded.card}`).
- Background: Panel white (`#ffffffcc`) over the field.
- Border: 1px Hairline, always. It does the separating now that shadows are gone.
- Shadow: none at rest (see Elevation).
- Padding: `1rem 1.05rem`.
- Banned: nested cards, a coloured stripe on any edge, a gradient bar across the top.

### Inputs

- Style: 18px radius, 1px border. On the light field: white fill, hairline border. On the console: white at 12% fill, white at 26% border, Console Ink text, and a visible caret and placeholder colour (never the inherited dark default).
- Focus: border turns Instrument Periwinkle and the focus ring appears. No glow.
- Error: border and helper text in Deep Red. The helper text says what to do, not only what broke.
- Disabled: 55% opacity, no hover response.

### Navigation

- Tabs: pill shaped (`999px`), sentence case, weight 600. Idle: panel white, hairline border, ink label. Selected: solid Ink fill, white label, a hard switch and not a tint. Hovering an idle tab darkens the hairline only. No blur, no glass.
- Sidebar: the console. Dark gradient (`#0d1320` to `#101925`), Console Ink labels, a hairline right border. It holds run settings and the data source: the levers, never the evidence.

### Console hero (signature component)

The dark banner at the top of every Lab app (`.editorial-hero` here, `.app-hero` in product-decision-under-uncertainty, `.mmf-hero` in measurement-maturity-framework). It is the one place the system raises its voice.

- 28px radius, `1.9rem 2.1rem` padding, console gradient with a periwinkle radial wash top left and a softer mint wash top right, Console shadow, 1px white-at-8% border.
- Contents, in order: at most one kicker (sentence case or small caps, never both uppercase and wide tracking on every screen), the Display title, and one subtitle line capped at 48rem.
- The three heroes differ only in copy and exact radial placement. Structure, radius, padding and shadow are the same, which is what makes them read as one family.

### Charts

- Series colours come from the shared ramp (section 2), in order, and always pair with a non-colour channel: markers, dash patterns or direct labels.
- Gridlines `#dde3ec` at hairline weight. Axis titles and ticks in Ink. No chart title when a section heading already says it.
- Charts sit flat in a hairline panel, not a floating card.
- Never use mint for a series that doesn't mean "passed".

## 6. Do's and don'ts

Do:

- Keep `theme-tokens.css` byte-identical across the three repos. Edit it in one, copy it verbatim to the other two (`product-decision-under-uncertainty/static/` and `measurement-maturity-framework/mmf/`). A token change is a three-repo change.
- Alias the shared tokens locally (`--app-*`, `--mmf-*`, `--blue`). Never redefine their values.
- Use Deep Periwinkle (`#3a56e0`), Deep Mint (`#0a7d5c`), Deep Amber (`#9a6a12`) and Deep Red (`#c43d33`) for coloured text on a light surface. The light forms are for fills and marks.
- Pair every colour-coded meaning with a label, shape or dash, especially guardrail pass/fail.
- Give every panel a 1px hairline (`#10131a14`) and no shadow at rest.
- Honour `prefers-reduced-motion`: the panel entrance and hover lift become instant. Keep transitions at 140 to 250ms.
- Keep prose to about 70 characters (62rem) even though the container is 1440px.
- Turn controls into solid white chips when they sit on the dark console.

Don't:

- Ship anything that looks like a raw Streamlit default: the stock red primary, unstyled widgets, an untended prototype.
- Put a shadow on a resting surface. `0 18px 48px` under every chart, table and metric is the failure this system fixes.
- Use `backdrop-filter: blur()` as decoration, on tabs or cards.
- Put a gradient bar across a metric card or a coloured stripe down any edge.
- Stack a tiny uppercase eyebrow above every section and metric. One per screen, in the hero, or none.
- Use gradient text (`background-clip: text`) anywhere.
- Use mint for anything that hasn't passed a check.
- Add a second font family, or a `clamp()` heading anywhere but the hero title.
- Let chart series drift off the shared ramp. Per-app palettes are how three apps stop looking like one product.
- Show a guardrail verdict, scenario or series in colour alone.
