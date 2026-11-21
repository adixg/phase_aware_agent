"""Joint empirical forecasting from FreeInference admission features."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from random import Random

HORIZON = "initial_user_turn"
TARGETS = ("remaining_llm_count", "future_tool_count", "remaining_observed_duration_ms")
FEATURES = ("harness", "model", "initial_prompt_estimate")
QUANTILES = (0.1, 0.5, 0.9)


def admission_features(snapshot, horizon):
    if horizon != HORIZON or snapshot["horizon"] != horizon or snapshot["event"] != "admission":
        raise ValueError("The fitted baseline supports initial-user-turn admission snapshots")
    features = snapshot["features"]
    availability = snapshot["feature_available_ms"]
    inputs = tuple(features[field] for field in FEATURES)
    if any(value is not None and (
        availability[field] is None or availability[field] > snapshot["cutoff_ms"]
    ) for field, value in zip(FEATURES, inputs)):
        raise ValueError("Forecast input was unavailable at admission")
    return inputs


@dataclass(frozen=True)
class Marginal:
    values: tuple
    mass: tuple
    weighted_sum: tuple
    half_pair_distance: float

    @classmethod
    def from_atoms(cls, atoms, index):
        weights = defaultdict(float)
        for row, weight in atoms:
            weights[row[index]] += weight
        values = tuple(sorted(weights))
        mass, weighted = [0.0], [0.0]
        half_pair_distance = 0.0
        for value in values:
            weight = weights[value]
            half_pair_distance += weight * (value * mass[-1] - weighted[-1])
            mass.append(mass[-1] + weight)
            weighted.append(weighted[-1] + weight * value)
        return cls(values, tuple(mass), tuple(weighted), half_pair_distance)

    def quantile(self, probability):
        if not 0 <= probability <= 1:
            raise ValueError("Quantile probability outside [0, 1]")
        index = bisect_left(self.mass, probability, lo=1) - 1
        return self.values[min(index, len(self.values) - 1)]

    def crps(self, outcome):
        index = bisect_right(self.values, outcome)
        absolute_error = (
            outcome * (2 * self.mass[index] - self.mass[-1])
            + self.weighted_sum[-1] - 2 * self.weighted_sum[index]
        )
        return max(0.0, absolute_error - self.half_pair_distance)


@dataclass(frozen=True)
class Forecast:
    horizon: str
    cutoff_ms: int
    quantiles: dict
    support: dict
    distribution: tuple

    def sample(self, count, *, seed):
        """Draw whole target tuples in TARGETS order."""
        rows, weights = zip(*self.distribution)
        return Random(seed).choices(rows, weights=weights, k=count)


class EmpiricalPredictor:
    """Interpolate whole empirical target tuples along a nested grouping hierarchy."""

    def __init__(self, rows, *, prompt_edges, strength, conditional=True):
        if not rows or strength <= 0 or tuple(sorted(set(prompt_edges))) != tuple(prompt_edges):
            raise ValueError("Expected training rows, positive smoothing, and sorted unique bucket edges")
        self.prompt_edges = tuple(prompt_edges)
        self.strength = strength
        self.conditional = conditional
        self.groups = defaultdict(list)
        for inputs, targets in rows:
            for key in self.group_keys(inputs):
                self.groups[key].append(tuple(targets))
        # Keep the cache per fit so learning-curve fits can be released.
        self.distribution_for_key = lru_cache(maxsize=None)(self._distribution_for_key)

    @classmethod
    def load(cls, path):
        artifact = json.loads(Path(path).read_text())
        if (artifact["schema_version"] != 1 or artifact["horizon"] != HORIZON
                or tuple(artifact["features"]) != FEATURES or tuple(artifact["targets"]) != TARGETS):
            raise ValueError("Unsupported empirical forecast artifact")
        return cls(artifact["rows"], prompt_edges=artifact["prompt_edges"],
                   strength=artifact["strength"])

    def group_keys(self, inputs):
        harness, model, prompt = inputs
        yield ()
        if not self.conditional or harness is None:
            return
        yield (harness,)
        if model is None:
            return
        yield (harness, model)
        if prompt is not None:
            yield (harness, model, bisect_right(self.prompt_edges, prompt))

    def _distribution_for_key(self, key):
        rows = self.groups[key]
        if key:
            parent, _, _ = self.distribution_for_key(key[:-1])
            parent_scale = self.strength / (len(rows) + self.strength)
            weights = defaultdict(float, {row: weight * parent_scale for row, weight in parent})
            local_weight = 1 / (len(rows) + self.strength)
        else:
            weights = defaultdict(float)
            local_weight = 1 / len(rows)
        for row in rows:
            weights[row] += local_weight
        total = sum(weights.values())
        atoms = tuple((row, weight / total) for row, weight in weights.items())
        marginals = tuple(Marginal.from_atoms(atoms, index) for index in range(len(TARGETS)))
        return atoms, marginals, len(rows)

    def distribution(self, inputs):
        key = next(key for key in reversed(tuple(self.group_keys(inputs))) if key in self.groups)
        atoms, marginals, count = self.distribution_for_key(key)
        return atoms, marginals, {"group_depth": len(key), "group_rows": count}

    def predict(self, snapshot, horizon=HORIZON):
        atoms, marginals, support = self.distribution(admission_features(snapshot, horizon))
        quantiles = {
            name: {p: marginal.quantile(p) for p in QUANTILES}
            for name, marginal in zip(TARGETS, marginals)
        }
        return Forecast(horizon, snapshot["cutoff_ms"], quantiles, support, atoms)
