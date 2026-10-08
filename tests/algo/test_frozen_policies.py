from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.frozen_policies import is_policy_frozen, parse_frozen_policies


def make_cfg(frozen_policies, num_policies=4):
    return SimpleNamespace(frozen_policies=frozen_policies, num_policies=num_policies)


class TestFrozenPolicies:
    @pytest.mark.parametrize(
        "frozen_policies, expected",
        [("", set()), (None, set()), ("0", {0}), ("0,1,2", {0, 1, 2}), (" 3 , 1 ", {1, 3})],
    )
    def test_parse(self, frozen_policies, expected):
        assert parse_frozen_policies(make_cfg(frozen_policies)) == expected

    def test_missing_argument_means_no_frozen_policies(self):
        # configs saved before the option was introduced do not have the attribute
        assert parse_frozen_policies(SimpleNamespace(num_policies=4)) == frozenset()

    @pytest.mark.parametrize("frozen_policies", ["a", "0,x", "4", "-1", "0,0", "1.5"])
    def test_parse_invalid(self, frozen_policies):
        with pytest.raises(ValueError):
            parse_frozen_policies(make_cfg(frozen_policies))

    def test_is_policy_frozen(self):
        cfg = make_cfg("0,2")
        assert [is_policy_frozen(cfg, p) for p in range(4)] == [True, False, True, False]

    def test_frozen_learner_skips_training_and_saving(self):
        learner = Learner.__new__(Learner)
        learner.frozen = True
        learner.policy_id = 0
        learner.cfg = SimpleNamespace(keep_checkpoints=2)
        learner.is_initialized = True
        learner._prepare_batch = MagicMock()
        learner._train = MagicMock()
        learner._get_checkpoint_dict = MagicMock()

        assert learner.train(MagicMock()) is None
        assert learner.save() is False
        assert learner.save_best(0, "reward", 1.0) is False
        learner.save_milestone()

        learner._prepare_batch.assert_not_called()
        learner._train.assert_not_called()
        learner._get_checkpoint_dict.assert_not_called()
