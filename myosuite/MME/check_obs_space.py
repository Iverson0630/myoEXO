#!/usr/bin/env python3
import argparse
import json
import sys

import gymnasium as gym
import numpy as np
from register_env import make_env

def _stats(name, value):
    if isinstance(value, np.ndarray):
        finite = np.isfinite(value)
        return {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value[finite].min()) if finite.any() else None,
            "max": float(value[finite].max()) if finite.any() else None,
            "nan": int(np.isnan(value).sum()),
            "inf": int(np.isinf(value).sum()),
        }
    return {"name": name, "type": type(value).__name__}


def _walk_space(space, obs, prefix="obs"):
    results = []
    if isinstance(space, gym.spaces.Box):
        results.append(_stats(prefix, np.asarray(obs)))
    elif isinstance(space, gym.spaces.Dict):
        for key, subspace in space.spaces.items():
            results.extend(_walk_space(subspace, obs.get(key), f"{prefix}.{key}"))
    elif isinstance(space, gym.spaces.Tuple):
        for i, subspace in enumerate(space.spaces):
            results.extend(_walk_space(subspace, obs[i], f"{prefix}[{i}]"))
    else:
        results.append({"name": prefix, "space_type": type(space).__name__})
    return results


def main():
    parser = argparse.ArgumentParser(description="Check obs vs observation_space.")
    parser.add_argument("--env-id", required=True, help="Gymnasium env id.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--print-obs",
        action="store_true",
        help="Print raw obs (may be large).",
    )
    args = parser.parse_args()

    env = make_env(args.env_id)
    obs, info = env.reset(seed=args.seed)

    space = env.observation_space
    print("observation_space:", space)
    print("contains(reset_obs):", space.contains(obs))

    stats = _walk_space(space, obs)
    print("obs_stats:", json.dumps(stats, indent=2))
   

    env.close()


if __name__ == "__main__":
    sys.exit(main())
