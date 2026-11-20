"""Interviewer key, Q05-Q08. Run with Python 3 and PyTorch on CPU."""

import torch
from torch.nn import functional as F


def sft_loss(logits, ids, assistant, valid):
    targets = ids[:, 1:]
    take = assistant[:, 1:] & valid[:, 1:] & valid[:, :-1]
    if not take.any():
        raise ValueError("No assistant targets")
    return F.cross_entropy(logits[:, :-1][take].float(), targets[take])


def grpo_loss(new, old, reward, group_id, mask):
    with torch.no_grad():
        reward = reward.float()
        advantage = torch.zeros_like(reward)
        for group in group_id.unique():
            rows = group_id == group
            values = reward[rows]
            advantage[rows] = (values - values.mean()) / (values.std(unbiased=False) + 1e-6)
    action_adv = advantage[:, None].expand_as(new)[mask]
    ratio = (new[mask].float() - old.detach()[mask].float()).exp()
    surrogate = torch.minimum(ratio * action_adv, ratio.clamp(0.8, 1.2) * action_adv)
    return -surrogate.mean()


@torch.no_grad()
def gae(r, v, next_v, bootstrap, carry, valid, gamma, lam):
    td = r + gamma * bootstrap * next_v - v
    out = torch.zeros_like(r)
    running = torch.zeros_like(r[:, 0])
    for t in reversed(range(r.shape[1])):
        running = torch.where(valid[:, t], td[:, t] + gamma * lam * carry[:, t] * running, 0.)
        out[:, t] = running
    return out


def complete_groups(events, G):
    pending = {}
    completed = set()
    for task, version, sample_id, reward in events:
        key = (task, version)
        if key in completed:
            continue
        group = pending.setdefault(key, {})
        group[sample_id] = reward
        if len(group) == G:
            rewards = [group[index] for index in range(G)]
            del pending[key]
            completed.add(key)
            yield task, version, rewards


def check_sft():
    torch.manual_seed(28)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 4, 5, 0, 0, 0]])
    valid = ids != 0
    assistant = torch.tensor([[0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0]]).bool()
    logits = torch.randn(2, 6, 6, requires_grad=True)
    selected = [(0, 1, 2), (0, 3, 4), (0, 4, 5), (1, 0, 4), (1, 1, 5)]
    logp = logits.log_softmax(-1)
    expected = -torch.stack([logp[b, t, token] for b, t, token in selected]).mean()
    loss = sft_loss(logits, ids, assistant, valid)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected_grad_rows = torch.zeros_like(valid)
    for b, t, _ in selected:
        expected_grad_rows[b, t] = True
    assert torch.equal(logits.grad.abs().sum(-1) > 0, expected_grad_rows)
    # A valid target immediately following left pad has no valid predicting position.
    try:
        sft_loss(logits[:1, :2], ids[:1, :2], torch.ones(1, 2, dtype=torch.bool), valid[:1, :2])
    except ValueError:
        return
    raise AssertionError("Empty target set must be handled")


def check_grpo():
    groups = torch.tensor([7, 9, 7, 9])
    rewards = torch.tensor([0., 1., 1., 1.], requires_grad=True)
    old = torch.zeros(4, 2, requires_grad=True)
    new = torch.tensor([[1.5, 42.], [1., 1.], [1.5, .6], [1., 1.]]).log().requires_grad_()
    mask = torch.tensor([[1, 0], [1, 0], [1, 1], [1, 0]]).bool()
    loss = grpo_loss(new, old, rewards, groups, mask)
    scale = .5 / (.5 + 1e-6)
    torch.testing.assert_close(loss, torch.tensor((1.5 - 1.2 - .6) * scale / 5))
    loss.backward()
    gradient = torch.tensor([[1.5, 0.], [0., 0.], [0., -.6], [0., 0.]]) * scale / 5
    torch.testing.assert_close(new.grad, gradient)
    assert old.grad is None and rewards.grad is None
    order = torch.tensor([1, 0, 3, 2])
    shuffled = grpo_loss(new[order], old[order], rewards[order], groups[order], mask[order])
    torch.testing.assert_close(shuffled, loss)


def check_gae():
    r = torch.tensor([[0., 1., 3., 13.], [7., 0., 0., 0.]])
    v = torch.tensor([[.2, .5, .7, 20.], [2., 1., 1., 1.]], requires_grad=True)
    nxt = torch.tensor([[.5, .8, 100., 4.], [1., 1., 1., 1.]])
    boot = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]]).bool()
    carry = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 1]]).bool()
    valid = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]]).bool()
    advantage = gae(r, v, nxt, boot, carry, valid, 1, .5)
    torch.testing.assert_close(advantage, torch.tensor([[.95, 1.3, 2.3, 0.], [5., 0., 0., 0.]]))
    assert not advantage.requires_grad
    one_step = gae(r, v, nxt, boot, carry, valid, 1, 0)
    torch.testing.assert_close(one_step[0, 0], torch.tensor(.3))


def check_groups():
    events = [
        ("q", 1, 1, 7), ("z", 1, 0, 8), ("q", 1, 1, 7), ("q", 2, 0, 9),
        ("q", 1, 0, 5), ("z", 1, 1, 6), ("q", 1, 0, 5), ("q", 1, 1, 7),
    ]
    expected = [("q", 1, [5, 7]), ("z", 1, [8, 6])]
    assert list(complete_groups(events, 2)) == expected
    assert list(complete_groups([], 1)) == []
    assert list(complete_groups([("q", 2, 0, 0)] * 2, 1)) == [("q", 2, [0])]


if __name__ == "__main__":
    for number, check in enumerate([check_sft, check_grpo, check_gae, check_groups], start=5):
        check()
        print(f"PASS Q{number:02d} {check.__name__}")
