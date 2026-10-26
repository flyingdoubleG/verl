# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.trainer_base as trainer_base
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu

PROMPT_LENS = [3, 2, 4, 1]
RESPONSE_LENS = [3, 2, 4, 1]
UIDS = [0, 0, 1, 1]
REWARDS = {"acc": [1.0, 0.0, 0.0, 1.0], "fmt": [0.5, 1.0, 1.0, 0.0]}
GDPO_KEYS = list(REWARDS)


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _FakeTQ:
    """In-memory stand-in for the TransferQueue key-value API used by the trainer."""

    def __init__(self, fields: dict):
        self.store = dict(fields)

    def kv_batch_get(self, keys, partition_id, select_fields):
        return tu.get_tensordict({k: self.store[k] for k in select_fields if k in self.store})

    def kv_batch_put(self, keys, partition_id, fields):
        for k in fields.keys():
            self.store[k] = fields[k]
        return SimpleNamespace(keys=keys, partition_id=partition_id, extra_info={})


class _Batch(SimpleNamespace):
    def __len__(self):
        return len(self.keys)


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _rm_scores():
    rows = []
    for i, r in enumerate(RESPONSE_LENS):
        row = torch.zeros(r)
        row[-1] = sum(REWARDS[key][i] for key in GDPO_KEYS)
        rows.append(row)
    return rows


def _make_trainer(colocated_reward: bool):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "gdpo",
                "gdpo_reward_keys": GDPO_KEYS,
                "gamma": 1.0,
                "lam": 1.0,
                "use_kl_in_reward": False,
                "norm_adv_by_std_in_grpo": True,
                "rollout_correction": None,
            },
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    trainer.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=None if colocated_reward else [MagicMock()]
    )
    return trainer


def _fake_tq(colocated_reward: bool):
    fields = {
        "uid": torch.tensor(UIDS),
        "response_mask": _nested([torch.ones(r, dtype=torch.long) for r in RESPONSE_LENS]),
        "rm_scores": _nested(_rm_scores()),
    }
    if colocated_reward:
        fields.update({key: np.array(values) for key, values in REWARDS.items()})
    else:
        fields["extra_fields"] = [
            {"reward_extra_info": {key: REWARDS[key][i] for key in GDPO_KEYS}} for i in range(len(UIDS))
        ]
    return _FakeTQ(fields)


def _v0_advantages():
    """GDPO advantages from the v0 trainer's padded [prompt | response] layout."""
    max_prompt, max_response = max(PROMPT_LENS), max(RESPONSE_LENS)
    prompts = torch.zeros(len(UIDS), max_prompt, dtype=torch.long)
    attention_mask = torch.zeros(len(UIDS), max_prompt + max_response, dtype=torch.long)
    response_mask = torch.zeros(len(UIDS), max_response, dtype=torch.long)
    token_level_rewards = torch.zeros(len(UIDS), max_response)
    for i, (p, r) in enumerate(zip(PROMPT_LENS, RESPONSE_LENS, strict=True)):
        attention_mask[i, max_prompt - p : max_prompt + r] = 1
        response_mask[i, :r] = 1
        token_level_rewards[i, :r] = _rm_scores()[i]
    data = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "token_level_rewards": token_level_rewards,
        },
        non_tensors={"uid": np.array(UIDS, dtype=object), **{k: np.array(v) for k, v in REWARDS.items()}},
    )
    config = OmegaConf.create({"gdpo_reward_keys": GDPO_KEYS})
    data = compute_advantage(data, adv_estimator="gdpo", num_repeat=2, config=config)
    return [data.batch["advantages"][i, :r] for i, r in enumerate(RESPONSE_LENS)]


@pytest.mark.parametrize("colocated_reward", [False, True])
def test_v1_gdpo_matches_v0(monkeypatch, colocated_reward):
    fake_tq = _fake_tq(colocated_reward)
    monkeypatch.setattr(trainer_base, "tq", fake_tq)
    trainer = _make_trainer(colocated_reward)
    batch = _Batch(keys=[str(i) for i in range(len(UIDS))], partition_id="train", extra_info={})

    trainer._compute_advantage(batch, metrics={})

    advantages = fake_tq.store["advantages"].unbind()
    expected = _v0_advantages()
    # GDPO gives each sample a non-trivial advantage here, so a zero or mismatched result would fail.
    assert all(row.abs().sum() > 0 for row in expected)
    for actual_row, expected_row in zip(advantages, expected, strict=True):
        torch.testing.assert_close(actual_row, expected_row)
