"""One token-level policy update. Vary replay temperature to diagnose mismatches."""

import argparse
import importlib
import math

import torch
from torch import nn
from torch.distributions import Categorical

USER, COMPARE, BOT, TOOL, LOW, HIGH, DONE, GET, SUBMIT, A, B = range(11)
NAMES = "USER COMPARE BOT TOOL LOW HIGH DONE GET SUBMIT A B".split()
GROUP = 16


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.word = nn.Embedding(len(NAMES), 32)
        self.pos = nn.Embedding(32, 32)
        self.block = nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True)
        self.head = nn.Linear(32, len(NAMES))

    def forward(self, ids):
        length = ids.shape[1]
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        hidden = self.word(ids) + self.pos(torch.arange(length))
        return self.head(self.block(hidden, src_mask=causal_mask))


@torch.no_grad()
def collect(model, winners):
    rows = torch.arange(len(winners))
    ids = torch.tensor([USER, COMPARE]).repeat(len(rows), 1)
    seen = torch.zeros(len(rows), 2, dtype=torch.bool)
    alive = torch.ones(len(rows), dtype=torch.bool)
    reward = torch.zeros(len(rows))
    records = []
    length = torch.full_like(rows, 2)
    for _ in range(3):
        ids = torch.cat((ids, torch.full_like(rows[:, None], BOT)), dim=1)
        for allowed in ([GET, SUBMIT], [A, B]):
            # This two-token grammar defines the actual behavior distribution.
            dist = Categorical(logits=model(ids)[:, -1, allowed])
            choice = dist.sample()
            records.append((ids.shape[1] - 1, allowed, choice, dist.log_prob(choice), alive.clone()))
            token = torch.tensor(allowed)[choice]
            ids = torch.cat((ids, token[:, None]), dim=1)
        op, vendor = ids[:, -2], ids[:, -1] - A
        query = alive & (op == GET)
        submit = alive & (op == SUBMIT)
        reward[submit] = (seen.all(1) & (vendor == winners))[submit].float()
        seen[rows[query], vendor[query]] = True
        quote = torch.where(vendor == winners, LOW, HIGH)
        observation = torch.where(query, quote, DONE)
        ids = torch.cat((ids, torch.full_like(rows[:, None], TOOL), observation[:, None]), dim=1)
        length += alive.long() * 5
        alive &= ~submit
    return ids, records, reward, length


def main(replay_temperature=1.0, implementation="solutions"):
    ops = importlib.import_module(implementation)
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = Decoder()
    # Eight groups alternate between two hidden tasks. Group members share a task.
    winners = (torch.arange(8) % 2).repeat_interleave(GROUP)
    ids, records, reward, length = collect(model, winners)
    dense_logits = model(ids)
    scores, old_scores, active = [], [], []
    for pos, allowed, choice, old, mask in records:
        dist = Categorical(logits=dense_logits[:, pos, allowed] / replay_temperature)
        scores.append(dist.log_prob(choice))
        old_scores.append(old)
        active.append(mask)
    logp = torch.stack(scores, dim=1)
    old = torch.stack(old_scores, dim=1)
    mask = torch.stack(active, dim=1)
    delta = (logp.detach() - old).masked_fill(~mask, 0.0)
    max_diff = delta.abs().max().item()
    if max_diff >= 1e-4:
        row, column = (int(i) for i in torch.unravel_index(delta.abs().argmax(), delta.shape))
        pos = records[column][0]
        prefix = " ".join(NAMES[i] for i in ids[row, :pos + 1].tolist())
        print(f"prefix={prefix} chosen={NAMES[ids[row, pos + 1]]}")
        print(f"old_logp={old[row, column].item():.5f} new_logp={logp[row, column].item():.5f} "
              f"ratio={delta[row, column].exp().item():.5f} max_logp_error={max_diff:.5f} "
              f"valid_action_tokens={int(mask.sum())} optimizer_steps=0")
        raise ValueError("Behavior and replay disagree before the first optimizer step.")
    adv = ops.group_adv(reward.reshape(-1, GROUP)).reshape(-1, 1).expand_as(logp)
    loss = ops.clip_loss(logp, old, adv, mask)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    opt.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    assert grad_norm > 0 and torch.isfinite(loss)
    opt.step()
    example = int(reward.argmax())
    print("trace:", " ".join(NAMES[i] for i in ids[example, :length[example]].tolist()))
    print(f"implementation={implementation} verified={int(reward.sum())}/{len(reward)} "
          f"valid_action_tokens={int(mask.sum())} "
          f"max_logp_error={max_diff:.2e} grad_norm={grad_norm:.4f} optimizer_steps=1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-temperature", type=float, default=1.0,
                        help="Temperature used only by the learner. Sampling always uses 1.")
    parser.add_argument("--implementation", choices=["solutions", "mine"], default="solutions",
                        help="Use your lab/mine.py group_adv and clip_loss for the update.")
    args = parser.parse_args()
    temperature = args.replay_temperature
    if not math.isfinite(temperature) or temperature <= 0:
        parser.error("--replay-temperature must be finite and positive")
    main(temperature, args.implementation)
