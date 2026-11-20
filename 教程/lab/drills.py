"""Use python lab/drills.py mine to grade lab/mine.py, or omit mine for the key."""

import importlib
import sys

import torch
from torch import nn


def check_logps(ops):
    # Tiny causal decoder. Token 3 is an external observation; 0 is padding.
    torch.manual_seed(11)
    embed, rnn, head = nn.Embedding(8, 5), nn.GRU(5, 6, batch_first=True), nn.Linear(6, 8)
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 4, 5, 0, 0]])
    assistant = torch.tensor([[0, 1, 0, 1, 1], [0, 1, 1, 0, 0]]).bool()
    logits = head(rnn(embed(ids))[0])
    logits.retain_grad()
    logps, mask = ops.action_logps(logits, ids, assistant, ids != 0)
    expected = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 0]]).bool()
    assert torch.equal(mask, expected)
    # Simulate behavior-side scoring one prefix at a time, then compare.
    for t in range(ids.shape[1] - 1):
        last = head(rnn(embed(ids[:, :t + 1]))[0][:, -1])
        expected_logp = last.log_softmax(-1).gather(1, ids[:, t + 1:t + 2]).squeeze(1)
        torch.testing.assert_close(logps[:, t], expected_logp)
    (-logps[mask].mean()).backward()
    assert logits.grad[:, :-1][~mask].abs().sum() == 0
    assert logits.grad[:, :-1][mask].abs().sum() > 0
    assert embed.weight.grad[3].abs().sum() > 0  # Observation conditions later choices.
    left_ids = torch.tensor([[0, 1, 4]])
    _, left_mask = ops.action_logps(torch.zeros(1, 3, 8), left_ids,
                                    torch.tensor([[0, 1, 1]]).bool(), left_ids != 0)
    assert torch.equal(left_mask, torch.tensor([[False, True]]))


def check_gae(ops):
    # t=0 continues, t=1 is collection cutoff, t=2 belongs to a fresh episode.
    reward = torch.tensor([[0., 1., 3.]])
    value = torch.tensor([[.2, .5, .7]])
    nxt = torch.tensor([[.5, .8, 100.]])
    boot = torch.tensor([[1, 1, 0]])
    carry = torch.tensor([[1, 0, 0]])
    result = ops.gae(reward, value, nxt, boot, carry, 1., .5)
    torch.testing.assert_close(result, torch.tensor([[.95, 1.3, 2.3]]))


def check_groups(ops):
    rewards = torch.tensor([[0., 0., 1., 1.], [1., 1., 1., 1.], [0., 0., 0., 0.]])
    result = ops.group_adv(rewards)
    torch.testing.assert_close(result, torch.tensor([[-1., -1., 1., 1.], [0.]*4, [0.]*4]),
                               atol=3e-6, rtol=0)


def check_clip(ops):
    new = torch.tensor([[1.5, .6, 1.5, .6, 1.]]).log().requires_grad_()
    old = torch.zeros_like(new, requires_grad=True)
    adv = torch.tensor([[1., 1., -1., -1., 100.]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 0]]).bool()
    loss = ops.clip_loss(new, old, adv, mask)
    torch.testing.assert_close(loss, torch.tensor(.125))
    loss.backward()
    torch.testing.assert_close(new.grad, torch.tensor([[0., -.15, .375, 0., 0.]]))
    assert old.grad is None and adv.grad is None
    try:
        ops.clip_loss(new, old, adv, ~torch.ones_like(mask))
    except ValueError:
        return
    raise AssertionError("Empty action batch must raise ValueError")


if __name__ == "__main__":
    ops = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "solutions")
    for check in (check_logps, check_gae, check_groups, check_clip):
        check(ops)
        print("PASS", check.__name__)
