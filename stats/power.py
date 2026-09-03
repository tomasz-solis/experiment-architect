"""Power, variance, duration, and compliance mathematics for experiment planning.

`frequentist.py` sizes a test on a conversion rate. This module covers the parts
of planning that a binary sample-size formula leaves out:

- continuous outcomes (spend, session length), where the sample requirement is
  driven by variance rather than by a base rate
- variance reduction by CUPED / regression adjustment, which buys the same thing
  as extra traffic but costs analysis time instead of calendar time (stratifying
  at assignment is the other lever, and is a randomisation change this module
  does not model)
- turning a sample requirement into a calendar date, including the ramp phase and
  the maturation window the last enrolled cohort still needs
- treatment intensity, where a stronger dose lowers the sample requirement until
  diminishing returns take over
- non-compliance: intention-to-treat as the primary estimand, and the complier
  effect (CACE/LATE) as a clearly-labelled secondary quantity
- simulation-based power for skewed metrics, where the normal approximation is
  the thing most likely to be lying
- how small a guardrail regression the test could actually have caught

The governing relationship behind most of it, for a two-arm test at a 50/50 split:

    n_per_arm = 2 * (z_alpha + z_beta)^2 * sigma^2 / delta^2

so the sample requirement scales with the square of the outcome's standard
deviation and inversely with the square of the effect you want to detect.
Doubling the noise costs four times the traffic; doubling the effect saves it.
"""

from __future__ import annotations

import math
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from scipy import stats

from config import ALPHA, DEFAULT_POWER

# Boos & Hughes-Oliver (2000) rule of thumb: the one-sample t-test keeps its
# nominal error rate once n exceeds roughly 25 * skewness^2. Used here as a
# per-arm floor to decide whether to trust the normal approximation on a
# skewed metric or fall back to simulation.
SKEW_SAFE_N_MULTIPLIER = 25.0

# Largest number of resampled observations held in memory at once during
# simulation. Keeps peak memory near 16 MB of float64 regardless of how large
# the requested arm size is.
_SIMULATION_CHUNK_CELLS = 2_000_000


class ContinuousSampleSize(TypedDict):
    """Sample requirement for a two-arm test on a continuous outcome."""

    n_total: int
    n_treatment: int
    n_control: int
    mde_absolute: float
    mde_relative: float | None
    sd_used: float
    allocation_cost: float
    variance_retained: float


class DurationPlan(TypedDict):
    """Calendar translation of a sample requirement."""

    ramp_days: int
    enrolment_days: int
    maturation_days: int
    total_days: int
    enrolment_weeks: float
    binding_constraint: str


class SimulatedPower(TypedDict):
    """Empirical power and false-positive rate from resampled historical data."""

    power: float
    false_positive_rate: float
    n_per_arm: int
    relative_lift: float
    iterations: int
    winsorise_cap: float | None
    calibrated: bool


class SkewDiagnostics(TypedDict):
    """How far a metric departs from the shape mean-based inference assumes."""

    skewness: float
    share_in_top_1_pct: float
    zero_share: float
    n_for_clt: int
    normal_approximation_safe: bool


class ComplianceResult(TypedDict):
    """Intention-to-treat effect plus the complier effect it implies."""

    itt: float
    itt_ci: tuple[float, float]
    take_up: float
    cace: float | None
    cace_ci: tuple[float, float] | None
    estimand_note: str


class IntensityOption(TypedDict):
    """One treatment dose, costed in sample and calendar time."""

    label: str
    expected_effect: float
    n_total: int
    total_days: int
    days_saved_vs_weakest: int
    marginal_days_per_effect_point: float | None


def _z(probability: float) -> float:
    """Standard normal quantile. Wraps scipy so the intent reads clearly."""
    return float(stats.norm.ppf(probability))


def allocation_cost(split_ratio: float) -> float:
    """Sample-size multiplier a non-even split costs, relative to 50/50.

    A two-arm test is most efficient at 50/50 when the arms have similar
    variance, because the standard error of the difference is driven by the
    *smaller* arm. The multiplier is ``(1/p + 1/(1-p)) / 4``: 1.00 at 50/50,
    1.19 at 70/30, 1.56 at 80/20, 2.78 at 90/10.

    An uneven split can still be the right business call, for limited treatment
    capacity, cost per treated user, or risk exposure, but it should be a
    decision made with the multiplier in view rather than a default.
    """
    if not 0 < split_ratio < 1:
        raise ValueError("Split ratio must be strictly between 0 and 1.")
    return ((1 / split_ratio) + (1 / (1 - split_ratio))) / 4


def cuped_variance_retained(rho: float) -> float:
    """Fraction of outcome variance left after adjusting on a pre-period covariate.

    CUPED (Deng et al., 2013) regresses the outcome on pre-experiment behaviour
    and analyses the residual. Randomisation makes the covariate independent of
    assignment, so removing the variance it explains leaves the effect estimate
    unbiased while shrinking its standard error. The residual variance is
    ``(1 - rho^2)`` of the original, and since sample size scales with variance,
    the sample requirement falls by the same factor.

    rho = 0.6 removes 36% of the required traffic. rho = 0.8 removes 64%.
    Estimate rho from a real pre-period; do not assume it.
    """
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1.")
    return 1 - rho**2


def estimate_cuped_rho(pre_period: pd.Series, outcome: pd.Series) -> float:
    """Correlation between a pre-period covariate and the outcome.

    This is the input CUPED's payoff depends on, and it is measurable before the
    test runs: take the same population, an earlier window for the covariate and
    a later one for the outcome. Metrics with a long stable history (spend,
    sessions) often reach 0.5-0.8. First-touch or activation metrics, where most
    units have no history, usually do not.
    """
    if len(pre_period) != len(outcome):
        raise ValueError("Pre-period and outcome series must be the same length.")
    if len(pre_period) < 3:
        raise ValueError("At least 3 paired observations are required to estimate rho.")

    paired = pd.DataFrame({"pre": pre_period, "post": outcome}).dropna()
    if len(paired) < 3:
        raise ValueError("At least 3 non-missing paired observations are required.")
    if paired["pre"].std() == 0 or paired["post"].std() == 0:
        raise ValueError("Cannot estimate rho when either series has zero variance.")

    return float(paired["pre"].corr(paired["post"]))


def sample_size_continuous(
    sd: float,
    mde_absolute: float,
    baseline_mean: float | None = None,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    split_ratio: float = 0.5,
    rho: float = 0.0,
) -> ContinuousSampleSize:
    """Sample requirement for detecting an absolute change in a mean.

    Four inputs decide the answer and all four are choices, not facts:

    - ``alpha``: the false-positive rate accepted
    - ``power``: the chance of detecting a real effect of exactly MDE size
    - ``mde_absolute``: the smallest effect worth acting on, set from the
      business decision *before* the feasible sample is known
    - ``sd``: the outcome's standard deviation, measured on the same window,
      grain, and population as the outcome itself

    ``rho`` applies the CUPED variance reduction, and ``split_ratio`` applies the
    allocation penalty. Both change the answer materially, so both are reported
    back in the result rather than folded silently into the total.
    """
    if sd <= 0:
        raise ValueError("Standard deviation must be greater than zero.")
    if mde_absolute <= 0:
        raise ValueError("MDE must be greater than zero.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1.")

    cost = allocation_cost(split_ratio)
    retained = cuped_variance_retained(rho)
    z_sum = _z(1 - alpha / 2) + _z(power)
    n_total = 4 * (z_sum**2) * (sd**2) * retained * cost / (mde_absolute**2)

    n_treatment = int(math.ceil(n_total * split_ratio))
    n_control = int(math.ceil(n_total * (1 - split_ratio)))

    relative: float | None = None
    if baseline_mean is not None and baseline_mean != 0:
        relative = mde_absolute / abs(baseline_mean)

    return {
        "n_total": int(math.ceil(n_total)),
        "n_treatment": n_treatment,
        "n_control": n_control,
        "mde_absolute": float(mde_absolute),
        "mde_relative": relative,
        "sd_used": float(sd),
        "allocation_cost": cost,
        "variance_retained": retained,
    }


def mde_from_sample_continuous(
    sd: float,
    n_total: int,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    split_ratio: float = 0.5,
    rho: float = 0.0,
) -> float:
    """Smallest absolute effect a given sample can detect. Inverse of the above.

    This is the honest direction to run the calculation in when traffic is fixed.
    If the answer comes back larger than the effect the business would act on,
    the test cannot answer the question, and inflating the MDE to match the
    traffic only hides that.
    """
    if n_total <= 0:
        raise ValueError("Total sample must be greater than zero.")
    if sd <= 0:
        raise ValueError("Standard deviation must be greater than zero.")

    cost = allocation_cost(split_ratio)
    retained = cuped_variance_retained(rho)
    z_sum = _z(1 - alpha / 2) + _z(power)
    return float(z_sum * math.sqrt(4 * (sd**2) * retained * cost / n_total))


def plan_duration(
    n_total: int,
    daily_new_eligible: float,
    maturation_days: int = 0,
    ramp_days: int = 0,
) -> DurationPlan:
    """Turn a sample requirement into a calendar plan.

    Three things separate this from ``n / daily_traffic``:

    - **Newly eligible units, not daily actives.** A returning user who was
      already randomised adds no sample. The enrolment rate is the rate at which
      units pass eligibility *for the first time*.
    - **The maturation window binds on the last cohort.** If the metric is
      30-day spend, the last user enrolled still needs 30 days. A 21-day
      enrolment on a 30-day metric is a 51-day experiment.
    - **The ramp is not the powered test.** A safety canary runs at an uneven
      split for engineering confidence; the powered clock starts when the final
      allocation does.
    """
    if n_total <= 0:
        raise ValueError("Total sample must be greater than zero.")
    if daily_new_eligible <= 0:
        raise ValueError("Daily newly-eligible units must be greater than zero.")
    if maturation_days < 0 or ramp_days < 0:
        raise ValueError("Maturation and ramp days cannot be negative.")

    enrolment_days = int(math.ceil(n_total / daily_new_eligible))
    total = ramp_days + enrolment_days + maturation_days

    if maturation_days >= enrolment_days and maturation_days > 0:
        binding = (
            "Waiting for the metric. The last people to join need longer than it takes to sign "
            "everyone up."
        )
    elif ramp_days >= enrolment_days and ramp_days > 0:
        binding = "The slow start. Ramping up costs more days than signing everyone up does."
    else:
        binding = "Sign-up rate. How fast new users reach the test is what sets the timeline."

    return {
        "ramp_days": ramp_days,
        "enrolment_days": enrolment_days,
        "maturation_days": maturation_days,
        "total_days": total,
        "enrolment_weeks": enrolment_days / 7,
        "binding_constraint": binding,
    }


def skew_diagnostics(values: pd.Series, n_per_arm: int | None = None) -> SkewDiagnostics:
    """Describe how far a metric is from the shape mean-based tests assume.

    Spend and inflow metrics are usually right-skewed: most units at zero, a thin
    tail carrying most of the total. The central limit theorem still rescues the
    *mean*, but how much sample it needs depends on the skew. The reported
    ``n_for_clt`` is the Boos & Hughes-Oliver floor, ``25 * skewness^2`` per arm,
    below which the t-test's nominal error rate should not be trusted and
    simulation is the better sizing method.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        raise ValueError("At least 3 non-missing values are required for skew diagnostics.")

    array = clean.to_numpy(dtype=float)
    skewness = float(stats.skew(array))
    total = float(array.sum())
    top_cut = float(np.quantile(array, 0.99))
    top_share = float(array[array >= top_cut].sum() / total) if total != 0 else float("nan")
    n_for_clt = int(math.ceil(SKEW_SAFE_N_MULTIPLIER * skewness**2))

    return {
        "skewness": skewness,
        "share_in_top_1_pct": top_share,
        "zero_share": float((array == 0).mean()),
        "n_for_clt": n_for_clt,
        "normal_approximation_safe": n_per_arm is not None and n_per_arm >= n_for_clt,
    }


def _welch_p_values(
    control: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    """Two-sided Welch p-values for stacked simulation draws (one row per draw)."""
    n_c = control.shape[1]
    n_t = treatment.shape[1]
    var_c = control.var(axis=1, ddof=1) / n_c
    var_t = treatment.var(axis=1, ddof=1) / n_t
    se = np.sqrt(var_c + var_t)
    diff = treatment.mean(axis=1) - control.mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, diff / se, 0.0)
        # Welch-Satterthwaite degrees of freedom: the reason this is not a plain
        # normal tail is that it stays honest when the arms have different spread.
        df = np.where(
            se > 0,
            (var_c + var_t) ** 2 / (var_c**2 / (n_c - 1) + var_t**2 / (n_t - 1)),
            1.0,
        )
    p_values: np.ndarray = 2 * stats.t.sf(np.abs(t_stat), df)
    return p_values


def simulate_power(
    values: pd.Series,
    relative_lift: float,
    n_per_arm: int,
    alpha: float = ALPHA,
    iterations: int = 400,
    winsorise_quantile: float | None = None,
    seed: int = 12345,
) -> SimulatedPower:
    """Empirical power: resample real outcomes, inject the lift, run the test.

    When the metric is heavily skewed, the closed-form sample size rests on an
    approximation that may not hold at the sample you actually have. Simulation
    replaces the assumption with a measurement: draw synthetic experiments from
    the real historical distribution, apply the effect you are trying to detect
    to the treatment arm, run the exact test you plan to run, and count how often
    it rejects.

    The zero-lift case is run alongside it. If the false-positive rate under no
    effect is not close to alpha, the analysis method is miscalibrated and the
    powered sample size from any formula is fiction. ``calibrated`` reports that
    check rather than leaving it to the reader.

    ``winsorise_quantile`` caps the tail *inside the simulation*, so the power
    number reflects the analysis you will actually run. Capping is a metric
    definition decided before launch; choosing it after seeing the results to
    make an effect significant is p-hacking under another name.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 10:
        raise ValueError("At least 10 historical observations are required to simulate.")
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2.")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if winsorise_quantile is not None and not 0 < winsorise_quantile <= 1:
        raise ValueError("Winsorise quantile must be between 0 and 1.")

    pool = clean.to_numpy(dtype=float)
    cap = float(np.quantile(pool, winsorise_quantile)) if winsorise_quantile is not None else None
    rng = np.random.default_rng(seed)

    chunk = max(1, min(iterations, _SIMULATION_CHUNK_CELLS // max(n_per_arm, 1)))
    rejections = 0
    false_positives = 0
    done = 0

    while done < iterations:
        rows = min(chunk, iterations - done)
        control = pool[rng.integers(0, len(pool), size=(rows, n_per_arm))]
        treated = pool[rng.integers(0, len(pool), size=(rows, n_per_arm))]
        null_arm = pool[rng.integers(0, len(pool), size=(rows, n_per_arm))]

        lifted = treated * (1 + relative_lift)
        if cap is not None:
            control = np.minimum(control, cap)
            lifted = np.minimum(lifted, cap)
            null_arm = np.minimum(null_arm, cap)

        rejections += int((_welch_p_values(control, lifted) < alpha).sum())
        false_positives += int((_welch_p_values(control, null_arm) < alpha).sum())
        done += rows

    fpr = false_positives / iterations
    return {
        "power": rejections / iterations,
        "false_positive_rate": fpr,
        "n_per_arm": n_per_arm,
        "relative_lift": relative_lift,
        "iterations": iterations,
        "winsorise_cap": cap,
        # Monte Carlo slack: with a few hundred iterations the observed rate
        # wobbles around alpha, so the tolerance is deliberately loose.
        "calibrated": abs(fpr - alpha) <= max(0.02, alpha * 0.6),
    }


def compliance_effects(
    itt: float,
    itt_standard_error: float,
    take_up_treatment: float,
    take_up_control: float = 0.0,
    alpha: float = ALPHA,
) -> ComplianceResult:
    """Intention-to-treat effect, and the complier effect it implies.

    Randomisation happens at assignment. Opting in happens afterwards and is
    partly *caused* by the treatment, which makes opt-in a post-treatment
    variable. Comparing opt-ins with non-opt-ins, or treatment opt-ins with the
    whole control arm, compares self-selected groups and measures motivation
    rather than the treatment.

    ITT, meaning every randomised unit analysed in its assigned arm, stays the
    primary estimand because it answers the decision: what happens if we ship this.

    The effect among compliers (CACE, also called LATE) is a different and
    legitimate quantity, recovered by dividing ITT by the take-up gap between
    arms, which is the Wald instrumental-variables estimator using assignment as
    the instrument. It requires the exclusion restriction, that assignment affects
    the outcome only through take-up, and no defiers. It is a secondary,
    clearly-labelled quantity, never the headline.
    """
    if itt_standard_error < 0:
        raise ValueError("Standard error cannot be negative.")
    for label, value in (("treatment", take_up_treatment), ("control", take_up_control)):
        if not 0 <= value <= 1:
            raise ValueError(f"Take-up in the {label} arm must be between 0 and 1.")

    z_crit = _z(1 - alpha / 2)
    itt_ci = (itt - z_crit * itt_standard_error, itt + z_crit * itt_standard_error)
    gap = take_up_treatment - take_up_control

    if gap <= 0:
        return {
            "itt": itt,
            "itt_ci": itt_ci,
            "take_up": take_up_treatment,
            "cace": None,
            "cace_ci": None,
            "estimand_note": (
                "Both groups took it up at the same rate, so there is no way to separate out the "
                "people your change persuaded. Report the effect across everyone offered it."
            ),
        }

    cace = itt / gap
    cace_ci = (itt_ci[0] / gap, itt_ci[1] / gap)

    if gap < 0.10:
        note = (
            f"Only {gap:.1%} more people took it up in one group than the other, so this figure "
            "blows a small handful of users up by a large factor. It is fragile and the range "
            "around it is wide. Lead with the number across everyone offered it."
        )
    else:
        note = (
            f"Worked out by dividing the overall effect by the {gap:.1%} difference in take-up. "
            "It only holds if being offered the change did nothing at all except make people take "
            "it up. Keep it secondary to the number across everyone offered it."
        )

    return {
        "itt": itt,
        "itt_ci": itt_ci,
        "take_up": take_up_treatment,
        "cace": cace,
        "cace_ci": cace_ci,
        "estimand_note": note,
    }


def guardrail_detectable_harm(
    baseline_rate: float,
    n_total: int,
    split_ratio: float = 0.5,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    """Smallest relative regression in a guardrail rate this test could catch.

    A guardrail that the test had no power to move is not evidence of safety, and
    reporting it as "held" overstates what the experiment showed. Rare guardrails
    (failed payments, complaints, compliance flags) need far more sample than
    the primary metric, so a test powered for a 3% lift in conversion routinely
    cannot rule out a meaningful increase in a 0.2% failure rate.

    Returned as a relative change so it reads on the same scale as the primary
    MDE.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("Guardrail baseline rate must be between 0 and 1.")
    if n_total <= 0:
        raise ValueError("Total sample must be greater than zero.")

    sd = math.sqrt(baseline_rate * (1 - baseline_rate))
    absolute = mde_from_sample_continuous(
        sd=sd,
        n_total=n_total,
        alpha=alpha,
        power=power,
        split_ratio=split_ratio,
    )
    return absolute / baseline_rate


def intensity_options(
    doses: list[tuple[str, float]],
    baseline: float,
    sd: float,
    daily_new_eligible: float,
    maturation_days: int = 0,
    alpha: float = ALPHA,
    power: float = DEFAULT_POWER,
    split_ratio: float = 0.5,
    rho: float = 0.0,
) -> list[IntensityOption]:
    """Cost each treatment dose in sample and calendar time.

    A stronger dose can produce a larger effect, and sample scales with the
    inverse square of the effect, so intensity buys time cheaply, for a while.
    Three checks keep it honest: is the strong version the one you would actually
    ship, what does it cost per treated user, and is the response still climbing?
    Effects rarely scale linearly with dose, so the marginal column here usually
    flattens, and a test of an unshippable dose answers a question nobody asked.

    ``doses`` are ``(label, expected relative effect)`` pairs supplied by the
    caller from prior evidence. This function costs the assumptions; it does not
    invent the response curve.
    """
    if not doses:
        raise ValueError("At least one dose is required.")
    if baseline == 0:
        raise ValueError("Baseline must be non-zero to convert a relative effect.")

    priced: list[IntensityOption] = []
    for label, effect in doses:
        if effect <= 0:
            raise ValueError(f"Dose '{label}': expected effect must be greater than zero.")
        size = sample_size_continuous(
            sd=sd,
            mde_absolute=abs(baseline) * effect,
            baseline_mean=baseline,
            alpha=alpha,
            power=power,
            split_ratio=split_ratio,
            rho=rho,
        )
        duration = plan_duration(
            n_total=size["n_total"],
            daily_new_eligible=daily_new_eligible,
            maturation_days=maturation_days,
        )
        priced.append(
            {
                "label": label,
                "expected_effect": effect,
                "n_total": size["n_total"],
                "total_days": duration["total_days"],
                "days_saved_vs_weakest": 0,
                "marginal_days_per_effect_point": None,
            }
        )

    ordered = sorted(priced, key=lambda row: row["expected_effect"])
    weakest_days = ordered[0]["total_days"]
    previous: IntensityOption | None = None
    for row in ordered:
        row["days_saved_vs_weakest"] = weakest_days - row["total_days"]
        if previous is not None:
            effect_step = (row["expected_effect"] - previous["expected_effect"]) * 100
            if effect_step > 0:
                row["marginal_days_per_effect_point"] = (
                    previous["total_days"] - row["total_days"]
                ) / effect_step
        previous = row
    return ordered


MetricLayer = Literal["conversion", "activation", "value_per_active", "value_per_randomised"]

# Each layer answers a different question, and only one of them is safe as the
# headline decision metric. "Value per active user" conditions on a state the
# treatment itself can change, which breaks the randomised comparison inside the
# filtered population. See `post_treatment_risk`.
METRIC_LAYERS: dict[MetricLayer, str] = {
    "conversion": (
        "Did they sign up? Safe to measure and hard to argue with, but it tells you how many, "
        "never how much."
    ),
    "activation": (
        "Did they get far enough to actually use it? Still counted across everyone who entered "
        "the test."
    ),
    "value_per_active": (
        "How much the active users were worth. Careful: your change decides who ends up active, "
        "so the two groups stop being like-for-like. Background colour, not the decision."
    ),
    "value_per_randomised": (
        "How much everyone who entered the test was worth, counting the people who did nothing "
        "as zero. This is the one to decide on, because it holds up even when your change moves "
        "who takes part."
    ),
}


def post_treatment_risk(metric_layer: MetricLayer, filters_on_post_state: bool) -> tuple[str, str]:
    """Classify whether the outcome definition breaks randomisation.

    Do not filter, segment, or define an outcome on anything measured after
    assignment that the treatment could have moved. The canonical failure is
    reporting net inflows only among "active investors" when the treatment
    changes who becomes an active investor: the arms stop being comparable
    populations and the effect can flip sign.

    The fix is to define the outcome over everyone randomised, letting
    non-participants enter as the zero that they are.

    Returns ``(status, explanation)`` where status is ``ok``, ``caution``, or
    ``fail``, matching the convention in ``stats/sanity.py``.
    """
    if metric_layer == "value_per_active" or filters_on_post_state:
        return (
            "fail",
            "You are only counting people who did something your change could have caused. That "
            "leaves you comparing two different kinds of people rather than two versions of the "
            "product, so the comparison is no longer fair and the answer can come out backwards. "
            "Count everyone who entered the test, with the people who did nothing as zero, and "
            "keep the per-participant view beside it as background.",
        )
    if metric_layer == "conversion":
        return (
            "caution",
            "Sign-ups are safe to measure, but they only tell you how many, never how much. If "
            "the decision is about money, put value per user next to it.",
        )
    return ("ok", METRIC_LAYERS[metric_layer])
