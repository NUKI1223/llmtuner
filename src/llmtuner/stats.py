"""Statistics for eval results. Standard library only.

Everything here works on plain lists of pass/fail values so it can be tested
against known answers without a store or a model.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

# Keep the bootstrap fast on large datasets: at most this many random draws.
_MAX_DRAWS = 5_000_000


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_dict(self) -> dict:
        return asdict(self)


def bootstrap_mean(
    values: list[float], n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> Interval:
    """Percentile bootstrap interval for the mean of `values`.

    The seed is fixed so the same results always give the same interval.
    """
    n = len(values)
    if n == 0:
        nan = float("nan")
        return Interval(nan, nan, nan)
    point = sum(values) / n
    n_boot = max(1000, min(n_boot, _MAX_DRAWS // n))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    alpha = (1 - level) / 2
    low = means[math.floor(alpha * (n_boot - 1))]
    high = means[math.ceil((1 - alpha) * (n_boot - 1))]
    return Interval(point, low, high)


def wilson_interval(n_passed: int, n: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a pass rate. Unlike the bootstrap it is not degenerate at 0% and 100%."""
    if n == 0:
        nan = float("nan")
        return Interval(nan, nan, nan)
    p = n_passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def pass_rate(passed: list[bool], z: float = 1.96) -> Interval:
    return wilson_interval(sum(1 for p in passed if p), len(passed), z)


def paired_detection_floor(n: int) -> float | None:
    """Smallest change in pass rate an exact two-sided McNemar test at 0.05 can show on `n` shared cases.

    At least 6 cases must change in one direction with none the other way
    (p = 2 / 2**6 = 0.031), so the floor is 6 / n. None when n < 6: no
    difference can be shown at all.
    """
    return None if n < 6 else 6 / n


def fisher_exact(pass_a: int, n_a: int, pass_b: int, n_b: int) -> float:
    """Two-sided Fisher exact test: could these two pass counts come from one pass rate?

    Used for one case run several times before and after a change. With 5 trials a side it
    only reaches 0.05 for near-total swings: 0 of 5 against 5 of 5 gives 0.008, 1 against 5
    gives 0.048, 2 against 5 gives 0.17.
    """
    if n_a <= 0 or n_b <= 0:
        return 1.0
    passes, total = pass_a + pass_b, n_a + n_b

    def prob(x: int) -> float:  # chance that side a holds x of the passes
        return math.comb(n_a, x) * math.comb(n_b, passes - x) / math.comb(total, passes)

    observed = prob(pass_a)
    tail = sum(p for x in range(max(0, passes - n_b), min(n_a, passes) + 1)
               if (p := prob(x)) <= observed * (1 + 1e-9))
    return min(1.0, tail)


def mcnemar_p(regressions: int, improvements: int) -> float:
    """Exact two-sided McNemar test on the cases where two runs disagree."""
    n = regressions + improvements
    if n == 0:
        return 1.0
    k = min(regressions, improvements)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class Paired:
    n: int
    baseline_rate: float
    candidate_rate: float
    diff: Interval
    regressions: list[str]
    improvements: list[str]
    p_value: float

    @property
    def measurable(self) -> bool:
        """True when the exact McNemar test rejects "no difference" at 0.05.

        The bootstrap interval on the difference is for display only.
        """
        return self.n > 0 and self.p_value < 0.05

    def as_dict(self) -> dict:
        data = asdict(self)
        data["measurable"] = self.measurable
        return data


def paired_compare(baseline: dict[str, bool], candidate: dict[str, bool], **kwargs) -> Paired:
    """Compare two runs on the cases they share, case by case."""
    ids = sorted(set(baseline) & set(candidate))
    n = len(ids)
    diffs = [float(candidate[i]) - float(baseline[i]) for i in ids]
    regressions = [i for i in ids if baseline[i] and not candidate[i]]
    improvements = [i for i in ids if candidate[i] and not baseline[i]]
    nan = float("nan")
    return Paired(
        n=n,
        baseline_rate=sum(baseline[i] for i in ids) / n if n else nan,
        candidate_rate=sum(candidate[i] for i in ids) / n if n else nan,
        diff=bootstrap_mean(diffs, **kwargs),
        regressions=regressions,
        improvements=improvements,
        p_value=mcnemar_p(len(regressions), len(improvements)),
    )


@dataclass(frozen=True)
class Agreement:
    """How well a grader matches human labels on the same outputs."""

    n: int
    both_pass: int
    human_pass_grader_fail: int
    human_fail_grader_pass: int
    both_fail: int
    agreement: float
    kappa: float | None
    catch_rate: float | None  # of the outputs a human failed, the share the grader failed
    false_fail_rate: float | None  # of the outputs a human passed, the share the grader failed

    def as_dict(self) -> dict:
        return asdict(self)


def agreement(human: list[bool], grader: list[bool]) -> Agreement:
    if len(human) != len(grader):
        raise ValueError("human and grader lists must be the same length")
    n = len(human)
    pp = sum(1 for h, g in zip(human, grader) if h and g)
    pf = sum(1 for h, g in zip(human, grader) if h and not g)
    fp = sum(1 for h, g in zip(human, grader) if not h and g)
    ff = n - pp - pf - fp
    if n == 0:
        return Agreement(0, 0, 0, 0, 0, float("nan"), None, None, None)
    observed = (pp + ff) / n
    human_pass, grader_pass = (pp + pf) / n, (pp + fp) / n
    expected = human_pass * grader_pass + (1 - human_pass) * (1 - grader_pass)
    # Kappa is undefined when chance agreement is already 1 (everyone says the same thing).
    kappa = None if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)
    return Agreement(
        n=n,
        both_pass=pp,
        human_pass_grader_fail=pf,
        human_fail_grader_pass=fp,
        both_fail=ff,
        agreement=observed,
        kappa=kappa,
        catch_rate=ff / (fp + ff) if (fp + ff) else None,
        false_fail_rate=pf / (pp + pf) if (pp + pf) else None,
    )
