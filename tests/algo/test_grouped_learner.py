"""
Tests of the hooks for models that hold several independent policies ("sample groups") in one network
(ActorCritic.sample_groups() etc.): one learner update of a grouped model must match the updates of
separate standard learners, each trained on the samples of its own group.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import Tensor, nn

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import EPISODIC, OBSERVER_ONLY_STATS
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import alloc_trajectory_tensors, traj_queue_size_bytes
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.model.actor_critic import ActorCritic, ActorCriticSharedWeights, default_make_actor_critic_func
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

GROUP_KEY = "group"
NUM_ACTIONS = 3
OBS_DIM = 4


class _ToyGroupedActorCritic(ActorCritic):
    """N standard models; every sample goes through the model of its group (obs["group"])."""

    def __init__(self, model_factory, obs_space: gym.spaces.Dict, action_space, cfg, num_groups: int):
        super().__init__(obs_space, action_space, cfg)
        sub_obs_space = gym.spaces.Dict({k: v for k, v in obs_space.spaces.items() if k != GROUP_KEY})
        self.subs = nn.ModuleList(
            [ActorCriticSharedWeights(model_factory, sub_obs_space, action_space, cfg) for _ in range(num_groups)]
        )
        self.encoders = self.subs[0].encoders
        self.num_sample_groups = num_groups

    def _groups(self, g: Tensor) -> Tensor:
        return g.reshape(-1).long().clamp(0, self.num_sample_groups - 1)

    def sample_groups(self, obs) -> Optional[Tensor]:
        return obs[GROUP_KEY].long()

    def normalize_returns(self, x: Tensor, groups: Optional[Tensor], denormalize: bool = False) -> None:
        for k, sub in enumerate(self.subs):
            m = groups == k
            if m.any():
                y = x[m]
                sub.returns_normalizer(y, denormalize=denormalize)
                x[m] = y

    def clip_grad_norm_(self, max_norm: float) -> None:
        for sub in self.subs:
            torch.nn.utils.clip_grad_norm_(sub.parameters(), max_norm)

    def _per_group(self, groups: Tensor, fn) -> List:
        """Calls fn(k, mask) for each non-empty group."""
        return [(m, fn(k, m)) for k in range(self.num_sample_groups) if (m := groups == k).any()]

    @staticmethod
    def _assemble(n: int, parts, like: Tensor) -> Tensor:
        out = None
        for m, t in parts:
            if out is None:
                out = torch.zeros((n,) + t.shape[1:], dtype=t.dtype, device=like.device)
            out[m] = t
        return out

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        g = self._groups(normalized_obs_dict[GROUP_KEY])
        obs = {k: v for k, v in normalized_obs_dict.items() if k != GROUP_KEY}
        parts = self._per_group(g, lambda k, m: self.subs[k].forward_head({key: v[m] for key, v in obs.items()}))
        x = self._assemble(len(g), parts, g)
        # carry the group index with the head output, as the RNN inputs of the learner are re-ordered
        return torch.cat([x, g.unsqueeze(1).float()], dim=1)

    def forward_core(self, head_output: Tensor, rnn_states):
        x, g = head_output[:, :-1], self._groups(head_output[:, -1])
        parts = self._per_group(g, lambda k, m: self.subs[k].forward_core(x[m], rnn_states[m]))
        out = self._assemble(len(g), [(m, r[0]) for m, r in parts], g)
        new_states = self._assemble(len(g), [(m, r[1]) for m, r in parts], g)
        return torch.cat([out, g.unsqueeze(1).float()], dim=1), new_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool, action_mask=None) -> TensorDict:
        x, g = core_output[:, :-1], self._groups(core_output[:, -1])
        parts = self._per_group(
            g, lambda k, m: self.subs[k].forward_tail(x[m], values_only=values_only, sample_actions=False)
        )
        result = TensorDict(values=self._assemble(len(g), [(m, r["values"].reshape(-1)) for m, r in parts], g))
        if values_only:
            return result
        logits = self._assemble(len(g), [(m, r["action_logits"]) for m, r in parts], g)
        result["action_logits"] = logits
        self.last_action_distribution = get_action_distribution(self.action_space, logits)
        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False, action_mask=None) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result


def _make_actor_critic(num_groups: int):
    def make(cfg, obs_space, action_space):
        if GROUP_KEY in obs_space.spaces:
            return _ToyGroupedActorCritic(global_model_factory(), obs_space, action_space, cfg, num_groups)
        return default_make_actor_critic_func(cfg, obs_space, action_space)

    return make


@pytest.fixture
def toy_model_factory():
    factory = global_model_factory()
    prev = factory.make_actor_critic_func
    yield lambda num_groups: factory.register_actor_critic_factory(_make_actor_critic(num_groups))
    factory.register_actor_critic_factory(prev)


def _env_info(with_group: bool) -> EnvInfo:
    spaces = {"obs": gym.spaces.Box(-10.0, 10.0, (OBS_DIM,), np.float32)}
    if with_group:
        spaces[GROUP_KEY] = gym.spaces.Box(0, 1000, (), np.float32)
    return EnvInfo(
        obs_space=gym.spaces.Dict(spaces),
        action_space=gym.spaces.Discrete(NUM_ACTIONS),
        num_agents=1,
        gpu_actions=False,
        gpu_observations=False,
        action_splits=[NUM_ACTIONS],
        all_discrete=True,
        frameskip=1,
    )


def _cfg(tmp_path, num_traj: int, rollout: int, **overrides):
    argv = [
        "--env=grouped_test",
        "--experiment=grouped_test",
        f"--train_dir={tmp_path}",
        "--device=cpu",
        "--serial_mode=True",
        f"--rollout={rollout}",
        f"--batch_size={num_traj * rollout}",
        "--num_batches_per_epoch=1",
        "--num_epochs=1",
        "--use_rnn=False",
        "--recurrence=1",
        "--with_vtrace=False",
        "--normalize_input=False",
        "--exploration_loss_coeff=0.003",
        "--encoder_mlp_layers",
        "16",
        "16",
        "--seed=0",
    ]
    parser, _ = parse_sf_args(argv)
    cfg = parse_full_cfg(parser, argv)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_learner(cfg, env_info: EnvInfo) -> Learner:
    policy_versions = torch.zeros([1], dtype=torch.int32)
    param_server = ParameterServer(0, policy_versions, cfg.serial_mode)
    learner = Learner(cfg, env_info, policy_versions, 0, param_server)
    learner.init()
    return learner


def _make_batch(cfg, env_info: EnvInfo, num_traj: int, groups: Tensor, invalid_steps: List[int]) -> TensorDict:
    """Synthetic trajectories; groups[i] is the (constant) group of trajectory i."""
    gen = torch.Generator().manual_seed(1)
    rollout = cfg.rollout
    b = alloc_trajectory_tensors(env_info, num_traj, rollout, get_rnn_size(cfg), torch.device("cpu"), False)
    b["obs"]["obs"][:] = torch.randn(b["obs"]["obs"].shape, generator=gen)
    if GROUP_KEY in b["obs"]:
        b["obs"][GROUP_KEY][:] = groups.float().unsqueeze(1)
    b["rnn_states"][:] = 0.0
    b["action_logits"][:] = torch.randn(b["action_logits"].shape, generator=gen)
    b["actions"][:] = torch.randint(0, NUM_ACTIONS, b["actions"].shape, generator=gen).float()
    log_p = torch.log_softmax(b["action_logits"], dim=-1)
    b["log_prob_actions"][:] = torch.gather(log_p, -1, b["actions"].long()).squeeze(-1)
    b["values"][:] = torch.randn(b["values"].shape, generator=gen)
    b["rewards"][:] = torch.randn(b["rewards"].shape, generator=gen)
    b["dones"][:] = torch.rand(b["dones"].shape, generator=gen) < 0.1
    b["time_outs"][:] = False
    b["policy_id"][:] = 0
    b["policy_id"][:, invalid_steps] = -1  # e.g. inactive agents; same steps in each trajectory
    b["policy_version"][:] = 0
    b["valids"][:] = True
    return b


def _drop_group(batch: TensorDict) -> TensorDict:
    res = TensorDict({k: v for k, v in batch.items() if k != "obs"})
    res["obs"] = TensorDict({k: v for k, v in batch["obs"].items() if k != GROUP_KEY})
    return res


def _max_diff(a: nn.Module, b: nn.Module) -> float:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    return max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa)


@pytest.mark.parametrize("normalize_returns", [False, True])
@pytest.mark.parametrize("max_grad_norm", [0.0, 4.0])
@pytest.mark.parametrize("invalid_steps", [[], [2, 3]])
def test_grouped_update_matches_separate_learners(
    tmp_path, toy_model_factory, normalize_returns, max_grad_norm, invalid_steps
):
    num_groups, rollout = 3, 8
    # unequal group sizes (5, 3, 4 trajectories) so that per-group means differ from the global mean
    groups = torch.tensor([0, 1, 2, 0, 0, 2, 1, 0, 2, 2, 0, 1])
    num_traj = len(groups)
    toy_model_factory(num_groups)

    cfg = _cfg(tmp_path, num_traj, rollout, normalize_returns=normalize_returns, max_grad_norm=max_grad_norm)
    grouped = _make_learner(cfg, _env_info(with_group=True))
    assert isinstance(grouped.actor_critic, _ToyGroupedActorCritic)
    batch = _make_batch(cfg, _env_info(with_group=True), num_traj, groups, invalid_steps)

    separate = []
    for k in range(num_groups):
        idx = (groups == k).nonzero().squeeze(1)
        cfg_k = _cfg(
            tmp_path / f"p{k}",
            len(idx),
            rollout,
            normalize_returns=normalize_returns,
            max_grad_norm=max_grad_norm,
        )
        learner = _make_learner(cfg_k, _env_info(with_group=False))
        learner.actor_critic.load_state_dict(grouped.actor_critic.subs[k].state_dict())
        learner.train(_drop_group(batch[idx]))
        separate.append(learner)

    grouped.train(batch)

    for k in range(num_groups):
        diff = _max_diff(grouped.actor_critic.subs[k], separate[k].actor_critic)
        assert diff < 1e-5, f"group {k}: max diff {diff}"


def test_other_groups_unchanged(tmp_path, toy_model_factory):
    num_groups, rollout = 3, 8
    groups = torch.zeros(6, dtype=torch.long)
    toy_model_factory(num_groups)
    cfg = _cfg(tmp_path, len(groups), rollout, normalize_returns=True, max_grad_norm=4.0)
    learner = _make_learner(cfg, _env_info(with_group=True))
    before = [{k: v.clone() for k, v in sub.state_dict().items()} for sub in learner.actor_critic.subs]
    learner.train(_make_batch(cfg, _env_info(with_group=True), len(groups), groups, []))
    after = [sub.state_dict() for sub in learner.actor_critic.subs]

    assert any(not torch.equal(before[0][k], after[0][k]) for k in before[0])
    for g in range(1, num_groups):
        for k in before[g]:
            assert torch.equal(before[g][k], after[g][k]), f"group {g} parameter {k} changed"


def test_default_hooks_are_standard(tmp_path):
    cfg = _cfg(tmp_path, 4, 8, normalize_returns=True)
    model = default_make_actor_critic_func(cfg, _env_info(False).obs_space, _env_info(False).action_space)
    assert model.num_sample_groups == 1
    assert model.sample_groups({"obs": torch.zeros(2, OBS_DIM)}) is None


def test_observer_only_stats_not_averaged():
    runner = AttrDict(cfg=AttrDict(stats_avg=100, num_policies=2), policy_avg_stats={})
    extra = {"cooperation": 0.5, OBSERVER_ONLY_STATS: {"policy": np.array([0, 1]), "episode_reward": np.array([1.0, 2.0])}}
    msg = {EPISODIC: {"reward": 1.0, "len": 10, "episode_extra_stats": extra}}
    Runner._episodic_stats_handler(runner, msg, 1)

    assert set(runner.policy_avg_stats) == {"reward", "len", "cooperation"}
    assert list(runner.policy_avg_stats["cooperation"][1]) == [0.5]
    # the message itself is not modified, so that registered handlers (observers) still receive everything
    assert OBSERVER_ONLY_STATS in msg[EPISODIC]["episode_extra_stats"]


def test_without_observer_only_stats_returns_same_object():
    d = {"reward": 1.0, "episode_extra_stats": {"a": 1}}
    assert Runner._without_observer_only_stats(d) is d


def test_traj_queue_size():
    assert traj_queue_size_bytes(1000) == 1_000_000
    assert traj_queue_size_bytes(51_200) > 1_000_000
    assert traj_queue_size_bytes(200_000) >= 64 * 200_000
