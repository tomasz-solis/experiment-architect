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

import json
import math
import re
from datetime import UTC, datetime
from typing import Literal, TypedDict, cast

from scipy.stats import chi2_contingency

from config import ALPHA, DEFAULT_POWER
from stats.frequentist import confidence_interval_binary
from stats.power import (
    METRIC_LAYERS,
    MetricLayer,
    guardrail_detectable_harm,
    mde_from_sample_continuous,
)

Status = Literal["ok", "caution", "fail"]

# Below this fraction of the planned sample, the test is treated as underpowered
# rather than merely short: at 90% of planned n the detectable effect grows by
# about 5%, which rarely changes a decision, and the gap widens quickly below that.
SAMPLE_SHORTFALL_TOLERANCE = 0.90

# Above this multiple of the planned sample, running long stops looking like
# ordinary overshoot. A quarter more than planned is what a test left running
# over a weekend picks up; a large overshoot instead suggests the stopping
# rule was not the one written down (e.g. "wait until it turns significant"),
# which inflates the false-positive rate the same way early peeking does.
OVER_DELIVERY_TOLERANCE = 1.25

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
    cluster_design_effect: float
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
    guardrail_approximation_valid: bool | None
    guardrail_higher_is_worse: bool
    decision_rule: str


# Every field a plan needs, and the JSON types that field is allowed to hold.
# ``parse_preregistration`` checks an upload against this table instead of
# trusting it, since a plan file can be hand-edited or come from an older,
# incompatible version of this app.
_PREREGISTRATION_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "created_at": (str,),
    "primary_metric": (str,),
    "metric_layer": (str,),
    "baseline": (int, float),
    "mde_relative": (int, float),
    "alpha": (int, float),
    "power": (int, float),
    "split_ratio": (int, float),
    "rho": (int, float),
    "cluster_design_effect": (int, float),
    "n_total": (int,),
    "ramp_days": (int,),
    "enrolment_days": (int,),
    "maturation_days": (int,),
    "total_days": (int,),
    "daily_new_eligible": (int, float),
    "transform": (str,),
    "estimand": (str,),
    "n_primary_metrics": (int,),
    "planned_looks": (int,),
    "guardrail_baseline": (int, float, type(None)),
    "guardrail_detectable_harm": (int, float, type(None)),
    "guardrail_approximation_valid": (bool, type(None)),
    "guardrail_higher_is_worse": (bool,),
    "decision_rule": (str,),
}

# Fields added after plans were already being written to disk. A plan missing
# one of these is not malformed, it is simply older than the field, so it gets
# this default instead of joining the missing-fields error. Applied before the
# missing-field check runs, so a genuinely absent field never reaches it, and
# a present-but-wrong-typed value is still caught by the type check above.
_COMPATIBILITY_DEFAULTS: dict[str, object] = {
    "cluster_design_effect": 1.0,
    "guardrail_higher_is_worse": True,
}


def serialise_preregistration(plan: PreRegistration) -> str:
    """Serialise a locked plan to indented, key-sorted JSON for download.

    Sorted keys and a fixed indent keep the file byte-stable across runs, so
    downloading the same plan twice produces an identical file.
    """
    return json.dumps(plan, indent=2, sort_keys=True)


def parse_preregistration(raw: str) -> PreRegistration:
    """Restore a plan from a previously downloaded file, validating as it goes.

    A restored file can be hand-edited, truncated, or left over from an
    incompatible version of this app, so this does not trust it: it checks
    the text is a JSON object, lists every missing field in one message
    instead of stopping at the first, and checks every field's type. Raises
    ``ValueError`` with a message a non-programmer can act on.

    A plan written before ``cluster_design_effect`` or
    ``guardrail_higher_is_worse`` existed is not treated as missing those
    fields: each gets the compatibility default in ``_COMPATIBILITY_DEFAULTS``
    (no clustering correction, higher-is-worse guardrail direction) instead of
    failing to load.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "This file is not valid JSON, so it cannot be read as a plan."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "This file is valid JSON, but a plan must be a single JSON object, not a "
            f"{type(payload).__name__}."
        )

    for key, default in _COMPATIBILITY_DEFAULTS.items():
        payload.setdefault(key, default)

    missing = sorted(key for key in _PREREGISTRATION_FIELD_TYPES if key not in payload)
    if missing:
        raise ValueError(
            "This file is missing fields a plan needs: " + ", ".join(missing) + "."
        )

    wrong_type = sorted(
        key
        for key, allowed_types in _PREREGISTRATION_FIELD_TYPES.items()
        if not isinstance(payload[key], allowed_types)
    )
    if wrong_type:
        raise ValueError(
            "These fields have the wrong kind of value: " + ", ".join(wrong_type) + "."
        )

    if payload["metric_layer"] not in METRIC_LAYERS:
        raise ValueError(
            f"'{payload['metric_layer']}' is not a metric type this app understands."
        )

    return cast(PreRegistration, payload)


class VerificationRow(TypedDict):
    """One pre-registered commitment, checked against what happened."""

    item: str
    planned: str
    actual: str
    status: Status
    note: str


class GuardrailReading(TypedDict):
    """What actually happened to a guardrail, read against the ship decision.

    A guardrail baseline and a detectable-harm figure only size the test. This
    is the other half: the reading the guardrail actually produced, so a
    launch decision can be checked against it instead of resting on the
    primary metric alone.
    """

    observed_control_rate: float
    observed_variant_rate: float
    relative_change: float
    ci_relative: tuple[float, float]
    detectable_harm: float | None
    status: Status
    note: str


class ReadoutSummary(TypedDict):
    """A result stated in the order a decision-maker can act on."""

    absolute_uplift: float
    relative_uplift: float
    ci_relative: tuple[float, float]
    business_impact: tuple[float, float] | None
    material: bool
    floor_clears_bar: bool
    conclusive: bool
    downside_ruled_out: bool
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
    cluster_design_effect: float = 1.0,
    transform: str = "none",
    estimand: str = "ITT",
    n_primary_metrics: int = 1,
    planned_looks: int = 1,
    guardrail_baseline: float | None = None,
    guardrail_higher_is_worse: bool = True,
    decision_rule: str = "",
) -> PreRegistration:
    """Freeze the design decisions, including the ones easiest to revise later.

    ``transform`` and ``n_primary_metrics`` are recorded here specifically
    because they are the two knobs most often turned after the data arrives.
    A winsorisation chosen once the tail is visible, or a third metric promoted
    to primary because the first two were flat, changes the false-positive rate
    of the whole exercise without leaving a trace unless it was written down.

    ``cluster_design_effect``, from :func:`stats.power.design_effect`, is
    recorded here so the sample requirement, the guardrail sensitivity, and
    later ``achieved_mde`` all stay sized under the same clustering
    assumption the plan was locked under. It defaults to 1.0, a no-op.
    """
    if not 0 < baseline < 1 and metric_layer in ("conversion", "activation"):
        raise ValueError("A rate baseline must be between 0 and 1.")
    if mde_relative <= 0:
        raise ValueError("MDE must be greater than zero.")
    if n_total <= 0:
        raise ValueError("Planned sample must be greater than zero.")
    if n_primary_metrics < 1:
        raise ValueError("There must be at least one primary metric.")
    if cluster_design_effect < 1.0:
        raise ValueError("Cluster design effect must be at least 1.0.")

    harm: float | None = None
    harm_valid: bool | None = None
    if guardrail_baseline is not None and 0 < guardrail_baseline < 1:
        guardrail = guardrail_detectable_harm(
            baseline_rate=guardrail_baseline,
            n_total=n_total,
            split_ratio=split_ratio,
            alpha=alpha,
            power=power,
            cluster_design_effect=cluster_design_effect,
        )
        harm = guardrail["relative_harm"]
        harm_valid = guardrail["approximation_valid"]

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
        "cluster_design_effect": cluster_design_effect,
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
        "guardrail_approximation_valid": harm_valid,
        "guardrail_higher_is_worse": guardrail_higher_is_worse,
        "decision_rule": decision_rule,
    }


def achieved_mde(plan: PreRegistration, actual_n_total: int) -> float:
    """Relative effect the delivered sample could actually detect.

    Recomputed from the sample that arrived, not the sample that was planned.
    When a test lands short, this is the number that says what a null result
    genuinely rules out. Applies the plan's ``cluster_design_effect`` so a
    delivered sample from a group-randomised test is not read as sharper than
    it actually is.
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
        cluster_design_effect=plan["cluster_design_effect"],
    )
    return absolute / baseline if baseline else float("nan")


def _normalise_metric_label(label: str) -> str:
    """Lowercase, strip, and collapse separators to a single space for comparison."""
    collapsed = re.sub(r"[_\-\s]+", " ", label.strip().lower())
    return collapsed.strip()


def metric_labels_match(planned: str, observed: str) -> bool:
    """Check whether a plan's free-text metric name plausibly names the observed one.

    The pre-registration stores the primary metric as free text, and a readout
    can be run on a dataset for a different metric entirely without anything
    catching it. This is a loose check, not a semantic one: either normalised
    label containing the other counts as a match, since analysts write the
    same metric several ways ("checkout conversion" vs "checkout conversion
    rate"). An empty label on either side has nothing to contradict, so it
    matches by default rather than raising a false alarm.
    """
    planned_norm = _normalise_metric_label(planned)
    observed_norm = _normalise_metric_label(observed)
    if not planned_norm or not observed_norm:
        return True
    return planned_norm in observed_norm or observed_norm in planned_norm


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
    if delivered > OVER_DELIVERY_TOLERANCE:
        sample_status: Status = "caution"
        sample_note = (
            "You collected well past the planned sample. That only stays honest if the stopping "
            "rule was fixed in advance: stopping once a result turned significant inflates the "
            "false-positive rate no matter how large the sample got."
        )
    elif delivered >= 1.0:
        sample_status = "ok"
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
        note = (
            f"The thing you must not break would have to get {harm:.1%} worse "
            "before this test noticed. If it looks untouched, that may only mean you "
            "could not see it: unverified, rather than clean."
        )
        if plan["guardrail_approximation_valid"] is False:
            note += (
                " There are too few events at this rate and sample size for this estimate to "
                "mean anything: use Fisher's exact test instead of trusting this number."
            )
        rows.append(
            {
                "item": "Guardrail sensitivity",
                "planned": f"detect ≥{harm:.1%} relative harm",
                "actual": f"{actual_n_total:,} units",
                "status": "caution" if harm > 0.10 else "ok",
                "note": note,
            }
        )

    return rows


def read_guardrail(
    control_events: int,
    control_n: int,
    variant_events: int,
    variant_n: int,
    detectable_harm: float | None = None,
    alpha: float = ALPHA,
    higher_is_worse: bool = True,
) -> GuardrailReading:
    """Check what a guardrail actually did, so a launch does not rest on the primary metric alone.

    A guardrail is only worth writing down if something later reads it. This is
    that reading: the observed rate in each arm, the interval on the relative
    change, and a status a decision-maker can act on without doing the
    interval maths themselves. ``higher_is_worse`` sets which direction is the
    harm: true (the default) matches how ``build_preregistration`` frames a
    guardrail as "the thing you must not break" (failed payments, complaints,
    and the like); false is for a guardrail where a drop is the harm instead
    (retention, successful deliveries), and flips the failing and cautionary
    conditions to the other side of zero.

    ``detectable_harm``, carried over from :func:`stats.power.guardrail_detectable_harm`,
    is what turns a clean reading into either "ok" or "caution": a guardrail
    that moved by less than the test could ever have noticed has not been
    verified as safe, it has simply not been checked, and the note says so
    rather than calling it proof.

    When the control arm has zero events, a relative change does not exist. Instead
    of raising an error, this function switches to absolute counts in the note and
    uses chi-squared test to determine whether the difference is statistically
    significant. Both arms zero (no events in either arm) returns status "caution"
    with note explaining neither arm was triggered. Control zero with variant events
    uses chi2_contingency to determine status based on the direction of harm.
    """
    for label, events, n in (("control", control_events, control_n), ("variant", variant_events, variant_n)):
        if n <= 0:
            raise ValueError(f"Guardrail {label} sample size must be greater than zero.")
        if events < 0:
            raise ValueError(f"Guardrail {label} event count cannot be negative.")
        if events > n:
            raise ValueError(f"Guardrail {label} event count cannot exceed the users measured.")

    control_rate = control_events / control_n
    variant_rate = variant_events / variant_n

    if control_rate <= 0:
        ci_relative: tuple[float, float] = (float("nan"), float("nan"))

        if variant_rate <= 0:
            relative_change = 0.0
            status: Status = "caution"
            note = (
                f"Neither arm saw this event (control: {control_events:,} of {control_n:,}; "
                f"variant: {variant_events:,} of {variant_n:,}). A guardrail that never fires "
                "in either arm cannot tell whether the change was safe."
            )
        else:
            relative_change = float("inf")
            chi2, p_value, dof, expected = chi2_contingency(
                [[control_events, control_n - control_events],
                 [variant_events, variant_n - variant_events]]
            )

            if higher_is_worse:
                if p_value < alpha:
                    status = "fail"
                    note = (
                        f"The variant saw {variant_events:,} of these in {variant_n:,} users "
                        f"where the control saw none ({control_events:,} of {control_n:,}). "
                        f"A percentage change against a zero baseline does not exist, but this "
                        f"difference is statistically significant (p={p_value:.4f}). This is new "
                        "harm appearing where there was none. Do not ship on the strength of the "
                        "primary result while this guardrail is failing."
                    )
                else:
                    status = "caution"
                    note = (
                        f"The variant saw {variant_events:,} of these in {variant_n:,} users "
                        f"where the control saw none ({control_events:,} of {control_n:,}). "
                        f"A percentage change against a zero baseline does not exist. This could "
                        f"be noise (p={p_value:.4f}), so watch it rather than calling it broken."
                    )
            else:
                if p_value < alpha:
                    status = "ok"
                    note = (
                        f"For this guardrail, more is improvement. The variant saw {variant_events:,} "
                        f"of these in {variant_n:,} users where the control saw none "
                        f"({control_events:,} of {control_n:,}). A percentage change against a "
                        f"zero baseline does not exist, but the absolute increase is statistically "
                        f"significant (p={p_value:.4f}) and represents improvement for this guardrail."
                    )
                else:
                    status = "caution"
                    note = (
                        f"For this guardrail, more is improvement. The variant saw {variant_events:,} "
                        f"of these in {variant_n:,} users where the control saw none "
                        f"({control_events:,} of {control_n:,}). A percentage change against a "
                        f"zero baseline does not exist. This increase could be noise (p={p_value:.4f})."
                    )
    else:
        relative_change = (variant_rate - control_rate) / control_rate
        ci_relative = confidence_interval_binary(control_rate, variant_rate, control_n, variant_n, alpha=alpha)
        ci_lower, ci_upper = ci_relative
        underpowered = detectable_harm is not None and abs(relative_change) < detectable_harm

        if higher_is_worse:
            failing = ci_lower > 0
            moved_the_wrong_way = relative_change > 0
        else:
            failing = ci_upper < 0
            moved_the_wrong_way = relative_change < 0

        if failing:
            status = "fail"
            zero_side = "stays above zero" if higher_is_worse else "stays below zero"
            note = (
                f"This moved {relative_change:+.1%}, and the range ({ci_lower:+.1%} to "
                f"{ci_upper:+.1%}) {zero_side}, so this is not noise. The ship decision does not "
                "belong to the primary metric alone: do not ship on the strength of the primary "
                "result while this guardrail is failing."
            )
        elif moved_the_wrong_way:
            status = "caution"
            note = (
                f"This moved {relative_change:+.1%}, the wrong way for this guardrail, but the "
                f"range ({ci_lower:+.1%} to {ci_upper:+.1%}) still touches zero, so this could be "
                "noise rather than real harm. Watch it rather than calling it broken or calling it "
                "clean."
            )
        elif underpowered:
            status = "caution"
            note = (
                f"This reads clean at {relative_change:+.1%}, but the test could only have caught "
                f"a problem of {detectable_harm:.1%} or bigger. A clean reading this small is "
                "unverified, not proof that nothing broke."
            )
        else:
            status = "ok"
            note = (
                f"This moved {relative_change:+.1%}, inside a range ({ci_lower:+.1%} to "
                f"{ci_upper:+.1%}) that does not point to harm."
            )

    return {
        "observed_control_rate": control_rate,
        "observed_variant_rate": variant_rate,
        "relative_change": relative_change,
        "ci_relative": ci_relative,
        "detectable_harm": detectable_harm,
        "status": status,
        "note": note,
    }


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
    the question, which is different from evidence of no effect. A result only
    earns that label once the interval rules out both a material gain and a
    material loss, or excludes zero outright; ruling out only the upside (a
    tight-looking upper bound while the lower bound still reaches a material
    loss) is not conclusive, it just means the test never checked for harm.

    ``material`` fires on the point estimate, so it can be true even when the
    pessimistic end of the range does not clear the bar. ``floor_clears_bar``
    is the stricter, interval-based version, and matches this app's own
    default decision rule: ship only if even the pessimistic end beats the bar.
    """
    if baseline == 0:
        raise ValueError("Baseline must be non-zero to express a relative uplift.")
    lower, upper = ci_relative
    if lower > upper:
        raise ValueError("Confidence interval lower bound exceeds the upper bound.")

    absolute = observed_rate_or_mean - baseline
    relative = absolute / baseline
    material = lower > 0 and relative >= mde_relative
    floor_clears_bar = lower >= mde_relative
    downside_ruled_out = lower > -mde_relative
    conclusive = lower > 0 or (upper < mde_relative and downside_ruled_out)

    impact: tuple[float, float] | None = None
    if population_size is not None and value_per_unit is not None:
        impact = (
            lower * baseline * population_size * value_per_unit,
            upper * baseline * population_size * value_per_unit,
        )

    if floor_clears_bar:
        headline = (
            f"Worth shipping: {relative:+.1%}, and even the pessimistic end of the range clears "
            f"the {mde_relative:.1%} bar."
        )
    elif material:
        headline = (
            f"Probably worth shipping: {relative:+.1%}, but the range runs as low as {lower:.1%}, "
            f"under the {mde_relative:.1%} bar."
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
    elif upper < mde_relative and not downside_ruled_out:
        headline = (
            f"Cannot rule out a loss: {relative:+.1%}, and the range runs down to {lower:.1%}. "
            "This test ruled out a win, not a loss."
        )
    else:
        headline = (
            f"Cannot tell: {relative:+.1%}, but the range runs from {lower:.1%} to {upper:.1%}. "
            "This test could not separate a win from a loss."
        )

    if upper < mde_relative and not downside_ruled_out:
        uncertainty_tail = (
            f"The upside is ruled out, but the bottom of the range is still a loss of "
            f"{abs(lower):.1%}, which this test has not ruled out."
        )
    elif upper < mde_relative:
        uncertainty_tail = (
            "Anything bigger than the top of that range is ruled out, so whatever is there is "
            "smaller than you care about."
        )
    else:
        uncertainty_tail = "That range still includes changes worth acting on."

    uncertainty = f"The true change is somewhere between {lower:.1%} and {upper:.1%}. " + uncertainty_tail

    return {
        "absolute_uplift": absolute,
        "relative_uplift": relative,
        "ci_relative": ci_relative,
        "business_impact": impact,
        "material": material,
        "floor_clears_bar": floor_clears_bar,
        "conclusive": conclusive,
        "downside_ruled_out": downside_ruled_out,
        "headline": headline,
        "uncertainty_line": uncertainty,
    }
