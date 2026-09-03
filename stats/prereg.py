"""Pre-registration contract, and the verification that carries it into the readout.

A design is only worth writing down if something later checks whether it was
honoured. This module holds both halves:

- ``build_preregistration`` freezes the decisions made before launch: MDE,
  alpha, power, split, sample, duration, the outcome transform, the estimand,
  and how many metrics count as primary
- ``verify_against_plan`` compares what actually happened against that record and
  returns one row per commitment, so a readout leads with "what we promised
  versus what we got" instead of a lift number with no provenance

Every row corresponds to a way an experiment silently stops answering the
question it was designed for: the sample landed short, the split drifted, the
last cohort never matured, a second metric became primary after the fact, or the
tail got trimmed once the result was visible.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal, TypedDict

from config import ALPHA, DEFAULT_POWER
from stats.power import MetricLayer, guardrail_detectable_harm, mde_from_sample_continuous

Status = Literal["ok", "caution", "fail"]

# Below this fraction of the planned sample, the test is treated as underpowered
# rather than merely short: at 90% of planned n the detectable effect grows by
# about 5%, which rarely changes a decision, and the gap widens quickly below that.
SAMPLE_SHORTFALL_TOLERANCE = 0.90

# Split drift beyond this is a sample-ratio-mismatch signal: a symptom of broken
# assignment, filtering, or logging, and a reason to stop reading the effect
# until the cause is known.
SPLIT_DRIFT_TOLERANCE = 0.02


class PreRegistration(TypedDict):
    """The frozen record of what was decided before the traffic was spent."""

    created_at: str
    primary_metric: str
    metric_layer: MetricLayer
    baseline: float
    mde_relative: float
    alpha: float
    power: float
    split_ratio: float
    rho: float
    n_total: int
    ramp_days: int
    enrolment_days: int
    maturation_days: int
    total_days: int
    daily_new_eligible: float
    transform: str
    estimand: str
    n_primary_metrics: int
    planned_looks: int
    guardrail_baseline: float | None
    guardrail_detectable_harm: float | None
    decision_rule: str


class VerificationRow(TypedDict):
    """One pre-registered commitment, checked against what happened."""

    item: str
    planned: str
    actual: str
    status: Status
    note: str


class ReadoutSummary(TypedDict):
    """A result stated in the order a decision-maker can act on."""

    absolute_uplift: float
    relative_uplift: float
    ci_relative: tuple[float, float]
    business_impact: tuple[float, float] | None
    material: bool
    conclusive: bool
    headline: str
    uncertainty_line: str


def build_preregistration(
    primary_metric: str,
    metric_layer: MetricLayer,
    baseline: float,
    mde_relative: float,
    n_total: int,
    ramp_days: int,
    enrolment_days: int,
    maturation_days: int,
    daily_new_eligible: float,
    split_ratio: float = 0.5,
    rho: float = 0.0,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    transform: str = "none",
    estimand: str = "ITT",
    n_primary_metrics: int = 1,
    planned_looks: int = 1,
    guardrail_baseline: float | None = None,
    decision_rule: str = "",
) -> PreRegistration:
    """Freeze the design decisions, including the ones easiest to revise later.

    ``transform`` and ``n_primary_metrics`` are recorded here specifically
    because they are the two knobs most often turned after the data arrives.
    A winsorisation chosen once the tail is visible, or a third metric promoted
    to primary because the first two were flat, changes the false-positive rate
    of the whole exercise without leaving a trace unless it was written down.
    """
    if not 0 < baseline < 1 and metric_layer in ("conversion", "activation"):
        raise ValueError("A rate baseline must be between 0 and 1.")
    if mde_relative <= 0:
        raise ValueError("MDE must be greater than zero.")
    if n_total <= 0:
        raise ValueError("Planned sample must be greater than zero.")
    if n_primary_metrics < 1:
        raise ValueError("There must be at least one primary metric.")

    harm: float | None = None
    if guardrail_baseline is not None and 0 < guardrail_baseline < 1:
        harm = guardrail_detectable_harm(
            baseline_rate=guardrail_baseline,
            n_total=n_total,
            split_ratio=split_ratio,
            alpha=alpha,
            power=power,
        )

    return {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "primary_metric": primary_metric,
        "metric_layer": metric_layer,
        "baseline": baseline,
        "mde_relative": mde_relative,
        "alpha": alpha,
        "power": power,
        "split_ratio": split_ratio,
        "rho": rho,
        "n_total": n_total,
        "ramp_days": ramp_days,
        "enrolment_days": enrolment_days,
        "maturation_days": maturation_days,
        "total_days": ramp_days + enrolment_days + maturation_days,
        "daily_new_eligible": daily_new_eligible,
        "transform": transform,
        "estimand": estimand,
        "n_primary_metrics": n_primary_metrics,
        "planned_looks": planned_looks,
        "guardrail_baseline": guardrail_baseline,
        "guardrail_detectable_harm": harm,
        "decision_rule": decision_rule,
    }


def achieved_mde(plan: PreRegistration, actual_n_total: int) -> float:
    """Relative effect the delivered sample could actually detect.

    Recomputed from the sample that arrived, not the sample that was planned.
    When a test lands short, this is the number that says what a null result
    genuinely rules out.
    """
    if actual_n_total <= 0:
        raise ValueError("Actual sample must be greater than zero.")
    baseline = plan["baseline"]
    sd = math.sqrt(baseline * (1 - baseline)) if 0 < baseline < 1 else baseline
    absolute = mde_from_sample_continuous(
        sd=sd,
        n_total=actual_n_total,
        alpha=plan["alpha"],
        power=plan["power"],
        split_ratio=plan["split_ratio"],
        rho=plan["rho"],
    )
    return absolute / baseline if baseline else float("nan")


def verify_against_plan(
    plan: PreRegistration,
    actual_n_total: int,
    actual_split_ratio: float,
    actual_days: int | None = None,
    metrics_tested: int = 1,
    looks_taken: int = 1,
    transform_applied: str = "none",
    analysed_as_itt: bool = True,
    maturation_complete: bool = True,
) -> list[VerificationRow]:
    """Check what happened against what was promised, one row per commitment.

    Ordered so the rows that invalidate the effect come before the rows that
    merely qualify it. A split mismatch or a broken estimand is a reason to stop
    and find the cause; a slightly short sample is a reason to widen the caveat.
    """
    rows: list[VerificationRow] = []

    delivered = actual_n_total / plan["n_total"]
    achieved = achieved_mde(plan, actual_n_total)
    if delivered >= 1.0:
        sample_status: Status = "ok"
        sample_note = (
            "You got all the users you planned for, so the change you set out to detect is still "
            "the change you can detect."
        )
    elif delivered >= SAMPLE_SHORTFALL_TOLERANCE:
        sample_status = "caution"
        sample_note = (
            f"A little short. The smallest change you can now spot is {achieved:.1%} rather than "
            f"the {plan['mde_relative']:.1%} you planned for."
        )
    else:
        sample_status = "fail"
        sample_note = (
            f"Well short of the plan. With these numbers the change would have to be "
            f"{achieved:.1%} or bigger before this test could see it, so a flat result here "
            "means you could not tell, not that nothing happened."
        )
    rows.append(
        {
            "item": "Sample delivered",
            "planned": f"{plan['n_total']:,}",
            "actual": f"{actual_n_total:,} ({delivered:.0%})",
            "status": sample_status,
            "note": sample_note,
        }
    )

    drift = abs(actual_split_ratio - plan["split_ratio"])
    rows.append(
        {
            "item": "Assignment split",
            "planned": f"{plan['split_ratio']:.0%} variant",
            "actual": f"{actual_split_ratio:.1%} variant",
            "status": "fail" if drift > SPLIT_DRIFT_TOLERANCE else "ok",
            "note": (
                "The two groups came out at different sizes to the split you set. Something is "
                "dropping users unevenly: how they are assigned, who qualifies, or what gets "
                "logged. Find out what before you read the result, because the groups may no "
                "longer be comparable."
                if drift > SPLIT_DRIFT_TOLERANCE
                else "Group sizes match the plan."
            ),
        }
    )

    rows.append(
        {
            "item": "Estimand",
            "planned": plan["estimand"],
            "actual": "ITT" if analysed_as_itt else "Not ITT",
            "status": "ok" if analysed_as_itt else "fail",
            "note": (
                "Everyone who entered the test is counted, in the group they were put in."
                if analysed_as_itt
                else "These numbers leave people out. Anyone who opted in or finished chose to do "
                "so, and that choice is not random, so what is left is a comparison of keen "
                "people against everyone. Recount across everyone who entered, in the group they "
                "were put in, and report the take-up effect separately."
            ),
        }
    )

    rows.append(
        {
            "item": "Outcome transform",
            "planned": plan["transform"],
            "actual": transform_applied,
            "status": "ok" if transform_applied == plan["transform"] else "fail",
            "note": (
                "The metric was trimmed the way the plan said."
                if transform_applied == plan["transform"]
                else "The metric was trimmed differently to the plan. Capping or cutting decided "
                "after you have seen the data is a way of picking the answer you like. Show both "
                "versions, and say plainly which one you agreed to first."
            ),
        }
    )

    if actual_days is not None:
        maturation_ok = maturation_complete and actual_days >= plan["total_days"]
        rows.append(
            {
                "item": "Duration and maturation",
                "planned": f"{plan['total_days']} days (incl. {plan['maturation_days']}d maturation)",
                "actual": f"{actual_days} days",
                "status": "ok" if maturation_ok else "caution",
                "note": (
                    "The last people to join got their full measurement window."
                    if maturation_ok
                    else "The people who joined last have not had their full measurement window "
                    "yet. They are being measured over less time than everyone else, which drags "
                    "the average down and usually hides a real effect. Wait, or leave them out."
                ),
            }
        )

    alpha_ok = metrics_tested <= plan["n_primary_metrics"]
    rows.append(
        {
            "item": "Primary metrics tested",
            "planned": str(plan["n_primary_metrics"]),
            "actual": str(metrics_tested),
            "status": "ok" if alpha_ok else "caution",
            "note": (
                "Matches the plan."
                if alpha_ok
                else f"You tested more metrics than you planned to. Across {metrics_tested} of "
                f"them, the chance that at least one looks like a winner on luck alone is "
                f"{1 - (1 - plan['alpha']) ** metrics_tested:.0%}. Either raise the bar or call "
                "the extra ones exploratory."
            ),
        }
    )

    looks_ok = looks_taken <= plan["planned_looks"]
    rows.append(
        {
            "item": "Interim looks",
            "planned": str(plan["planned_looks"]),
            "actual": str(looks_taken),
            "status": "ok" if looks_ok else "caution",
            "note": (
                "No extra peeking."
                if looks_ok
                else "Every extra early look raises the odds of a false alarm, because stopping "
                "the moment a number looks good is a way of catching lucky moments. Without a "
                "plan for that set in advance, treat this result as more flattering than it is."
            ),
        }
    )

    if plan["guardrail_detectable_harm"] is not None:
        harm = plan["guardrail_detectable_harm"]
        rows.append(
            {
                "item": "Guardrail sensitivity",
                "planned": f"detect ≥{harm:.1%} relative harm",
                "actual": f"{actual_n_total:,} units",
                "status": "caution" if harm > 0.10 else "ok",
                "note": (
                    f"The thing you must not break would have to get {harm:.1%} worse "
                    "before this test noticed. If it looks untouched, that may only mean you "
                    "could not see it: unverified, rather than clean."
                ),
            }
        )

    return rows


def summarise_readout(
    baseline: float,
    observed_rate_or_mean: float,
    ci_relative: tuple[float, float],
    mde_relative: float,
    population_size: int | None = None,
    value_per_unit: float | None = None,
) -> ReadoutSummary:
    """State a result in the order a decision-maker can act on.

    Decision and effect first, uncertainty second, business impact third. A
    p-value answers "would noise alone produce this?", which is a screening
    question, not the decision. Two results with identical p-values can carry
    opposite recommendations once the interval is read against the smallest
    effect worth acting on.

    ``conclusive`` is the distinction most readouts blur: an interval that spans
    both a meaningful gain and a meaningful loss is a test that failed to answer
    the question, which is different from evidence of no effect.
    """
    if baseline == 0:
        raise ValueError("Baseline must be non-zero to express a relative uplift.")
    lower, upper = ci_relative
    if lower > upper:
        raise ValueError("Confidence interval lower bound exceeds the upper bound.")

    absolute = observed_rate_or_mean - baseline
    relative = absolute / baseline
    material = lower > 0 and relative >= mde_relative
    conclusive = lower > 0 or upper < mde_relative

    impact: tuple[float, float] | None = None
    if population_size is not None and value_per_unit is not None:
        impact = (
            lower * baseline * population_size * value_per_unit,
            upper * baseline * population_size * value_per_unit,
        )

    if material:
        headline = (
            f"Worth shipping: {relative:+.1%}, above the {mde_relative:.1%} you set as worth "
            "acting on."
        )
    elif lower > 0:
        headline = (
            f"Real, but too small to act on: {relative:+.1%}, under the {mde_relative:.1%} you "
            "set as the bar."
        )
    elif upper < 0:
        headline = f"It made things worse: {relative:+.1%}, and the whole range sits below zero."
    elif conclusive:
        headline = (
            f"Nothing worth acting on: {relative:+.1%}, and anything bigger than {upper:.1%} is "
            "ruled out."
        )
    else:
        headline = (
            f"Cannot tell: {relative:+.1%}, but the range runs from {lower:.1%} to {upper:.1%}. "
            "This test could not separate a win from a loss."
        )

    uncertainty = (
        f"The true change is somewhere between {lower:.1%} and {upper:.1%}. "
        + (
            "Anything bigger than the top of that range is ruled out, so whatever is there is "
            "smaller than you care about."
            if upper < mde_relative
            else "That range still includes changes worth acting on."
        )
    )

    return {
        "absolute_uplift": absolute,
        "relative_uplift": relative,
        "ci_relative": ci_relative,
        "business_impact": impact,
        "material": material,
        "conclusive": conclusive,
        "headline": headline,
        "uncertainty_line": uncertainty,
    }
