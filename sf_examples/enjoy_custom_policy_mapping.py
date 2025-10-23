import sys

from sample_factory.enjoy import enjoy
from sample_factory.algo.utils.agent_policy_mapping import register_agent_policy_mapping
from sf_examples.train_custom_policy_mapping import parse_custom_args, register_custom_components


class SelfPlayPolicyMapping:
    """
    The simple mapping that assigns the same policy to all agents.
    """

    def __init__(self, policy_id, cfg, env_info):
        self.policy_id = policy_id
        self.num_policies = cfg.num_policies
        assert self.policy_id < self.num_policies, f"{self.policy_id=} must be less than {self.num_policies=}"

    def get_policy_for_agent(self, agent_idx: int, env_idx: int, global_env_idx: int) -> int:
        return self.policy_id


def make_self_play_policy_mapping_fn(policy_id):
    return lambda cfg, env_info: SelfPlayPolicyMapping(policy_id, cfg, env_info)


def main():
    """Script entry point."""
    register_custom_components()
    register_agent_policy_mapping("self_play_0", make_self_play_policy_mapping_fn(0))
    register_agent_policy_mapping("self_play_1", make_self_play_policy_mapping_fn(1))
    cfg = parse_custom_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
