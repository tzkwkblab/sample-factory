from typing import FrozenSet

from sample_factory.utils.typing import Config, PolicyID


def parse_frozen_policies(cfg: Config) -> FrozenSet[PolicyID]:
    """
    Parse --frozen_policies (comma-separated policy IDs, e.g. "0,1,2") into a set of policy IDs.
    Frozen policies are used for inference only: their learners skip all training and never save checkpoints.

    Raises ValueError if the value is malformed, contains duplicates or IDs outside [0, num_policies - 1].
    """
    frozen_str = getattr(cfg, "frozen_policies", "") or ""
    tokens = [token.strip() for token in str(frozen_str).split(",") if token.strip()]

    frozen = []
    for token in tokens:
        try:
            policy_id = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid policy id {token!r} in --frozen_policies={frozen_str!r}") from exc

        if policy_id < 0 or policy_id >= cfg.num_policies:
            raise ValueError(
                f"Invalid policy id {policy_id} in --frozen_policies={frozen_str!r}. "
                f"Valid range is [0, {cfg.num_policies - 1}]"
            )
        if policy_id in frozen:
            raise ValueError(f"Duplicate policy id {policy_id} in --frozen_policies={frozen_str!r}")
        frozen.append(policy_id)

    return frozenset(frozen)


def is_policy_frozen(cfg: Config, policy_id: PolicyID) -> bool:
    return policy_id in parse_frozen_policies(cfg)
