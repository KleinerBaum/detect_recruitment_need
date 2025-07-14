"""Reinforcement learning helpers for the Vacalyser wizard."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, List
import numpy as np
import yaml  # type: ignore

gym: Any
try:  # pragma: no cover - optional dependency
    import gymnasium as gym
except Exception:
    try:
        import gym  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional dependency missing
        gym = None

if gym is None:  # pragma: no cover - define placeholder
    GymEnv = object  # type: ignore[misc, assignment]
else:
    GymEnv = gym.Env  # type: ignore[misc, assignment]


if gym is not None:
    BaseEnv = gym.Env
else:

    class BaseEnv:  # pragma: no cover - minimal stub for missing gymnasium
        pass


# ---------------------------------------------------------------------------
# Schema Loading & State Vector
# ---------------------------------------------------------------------------


def load_wizard_schema(path: str) -> dict:
    """Load wizard step schema from a YAML file.

    Parameters
    ----------
    path:
        File path to the YAML schema.

    Returns
    -------
    dict
        Parsed schema dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def state_to_vector(session_state: dict[str, Any], schema: dict) -> np.ndarray:
    """Convert session state into a numerical vector.

    The vector contains the current step index followed by booleans for each
    defined field indicating whether it is present in ``session_state``.

    Parameters
    ----------
    session_state:
        Current session state values.
    schema:
        Wizard schema as returned by :func:`load_wizard_schema`.

    Returns
    -------
    np.ndarray
        Feature vector representation.
    """
    steps: List[dict] = schema.get("steps", [])
    step_names = [s["name"] for s in steps]
    current = session_state.get("wizard_step", step_names[0])
    step_index = step_names.index(current) if current in step_names else 0
    fields: List[str] = []
    for step in steps:
        for field in step.get("fields", []):
            fields.append(field["key"])
    vec = [float(step_index)]
    for key in fields:
        vec.append(float(bool(session_state.get(key))))
    return np.asarray(vec, dtype=float)


# ---------------------------------------------------------------------------
# Policy Classes
# ---------------------------------------------------------------------------


@dataclass
class WizardPolicy:
    """Simple policy deciding the next step."""

    step_order: List[str]

    def decide_next_step(self, state: dict[str, Any]) -> int:
        """Return index of the next step to visit."""
        current = state.get("wizard_step", self.step_order[0])
        try:
            idx = self.step_order.index(current)
        except ValueError:  # pragma: no cover - invalid state
            idx = 0
        return min(idx + 1, len(self.step_order) - 1)


def heuristic_next_step(session_state: dict[str, Any], schema: dict) -> int:
    """Baseline heuristic for next step decision."""
    policy = WizardPolicy([s["name"] for s in schema.get("steps", [])])
    return policy.decide_next_step(session_state)


# ---------------------------------------------------------------------------
# Reward & Environment
# ---------------------------------------------------------------------------


def compute_reward(session_metrics: dict[str, Any]) -> float:
    """Compute reward from session metrics."""
    steps = session_metrics.get("total_steps", 0)
    duration = session_metrics.get("total_time_sec", 0.0)
    completed = session_metrics.get("completed", False)
    reward = -steps - duration * 0.1
    if completed:
        reward += 10.0
    else:
        reward -= 5.0
    return reward


BaseEnv: type = gym.Env if gym is not None else object


class VacalyserWizardEnv(BaseEnv):  # type: ignore[misc]
    """Gym environment simulating the wizard."""

    def __init__(self, schema: dict) -> None:
        self.schema = schema
        self.step_order = [s["name"] for s in schema.get("steps", [])]
        if gym is not None:
            self.action_space = gym.spaces.Discrete(2)
            obs_len = 1 + sum(
                len(s.get("fields", [])) for s in self.schema.get("steps", [])
            )
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_len,))
        self.state: dict[str, Any] = {}


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            gym.Env.reset(self, seed=seed)  # type: ignore[attr-defined]
        self.state = {"wizard_step": self.step_order[0]}
        return state_to_vector(self.state, self.schema), {}

    def step(self, action: int):
        if gym is None:  # pragma: no cover - import guard
            raise ImportError("gymnasium not installed")
        idx = self.step_order.index(self.state.get("wizard_step"))
        if action == 1:  # skip
            idx = min(idx + 2, len(self.step_order) - 1)
        else:
            idx = min(idx + 1, len(self.step_order) - 1)
        self.state["wizard_step"] = self.step_order[idx]
        done = idx == len(self.step_order) - 1
        reward = 1.0 if done else 0.0
        return (
            state_to_vector(self.state, self.schema),
            reward,
            done,
            False,
            {},
        )


# ---------------------------------------------------------------------------
# Persistence Utilities
# ---------------------------------------------------------------------------


def save_policy(policy: WizardPolicy, filepath: str | Path) -> None:
    """Serialize policy to disk using pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(policy, f)


def load_policy(filepath: str | Path) -> WizardPolicy:
    """Load a pickled policy from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
