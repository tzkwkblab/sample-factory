"""
From the root of Sample Factory repo this can be run as:
python -m sf_examples.train_custom_policy_mapping --env=my_custom_multi_env_v1 --experiment=custom_mapping --train_dir=./train_dir --agent_policy_mapping=my_custom_mapping --num_policies=2

After training for a desired period of time, evaluate the policy by running:
python -m sf_examples.enjoy_custom_policy_mapping --env=my_custom_multi_env_v1 --experiment=custom_mapping --no_render --agent_policy_mapping=my_custom_mapping

Alternatively, you can evaluate the self-play of each policy by running:
python -m sf_examples.enjoy_custom_policy_mapping --env=my_custom_multi_env_v1 --experiment=custom_mapping --no_render --agent_policy_mapping=self_play_0
python -m sf_examples.enjoy_custom_policy_mapping --env=my_custom_multi_env_v1 --experiment=custom_mapping --no_render --agent_policy_mapping=self_play_1

"""

from __future__ import annotations

import sys

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.algo.utils.agent_policy_mapping import register_agent_policy_mapping
from sf_examples.train_custom_multi_env import make_custom_multi_env_func, add_extra_params_func
from sf_examples.train_custom_env_custom_model import make_custom_encoder, override_default_params


class CustomAgentPolicyMapping:
    """
    The simple mapping that assigns a different policy to each agent.
    """

    def __init__(self, cfg, env_info):
        self.num_agents = env_info.num_agents
        self.num_policies = cfg.num_policies
        assert self.num_agents == self.num_policies, f"{self.num_agents=} must be equal to {self.num_policies=}"

    def get_policy_for_agent(self, agent_idx: int, env_idx: int, global_env_idx: int) -> int:
        return agent_idx


def register_custom_components():
    register_env("my_custom_multi_env_v1", make_custom_multi_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)
    register_agent_policy_mapping("my_custom_mapping", CustomAgentPolicyMapping)


def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
