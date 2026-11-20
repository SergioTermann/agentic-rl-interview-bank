"""CPU lab for a two-vendor quotation agent. Run python lab/train.py --help."""

import argparse
import copy
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical


HORIZON = 3
FEATURES = 7
METHODS = ("rule", "random", "sft", "grpo", "sft-grpo", "sft-proxy")


def make_tasks(split, count):
    """Assign each tuple to one split by content. OOD also changes freight."""
    bucket = {"train": 0, "dev": 1, "test": 2, "shift": 3}[split]
    rng = random.Random(9301 + bucket)
    freight_max = 20 if split == "shift" else 5
    tasks = []
    seen = set()
    while len(tasks) < count:
        row = (rng.randint(5, 30), rng.randint(0, freight_max),
               rng.randint(5, 30), rng.randint(0, freight_max))
        # Keep both a pair and its vendor-swapped version in the same split.
        key = tuple(sorted((row[:2], row[2:])))
        if key in seen or sum(row[:2]) == sum(row[2:]):
            continue
        digest = hashlib.sha256(repr(key).encode("ascii")).digest()
        if digest[0] % 4 == bucket:
            tasks.append(row)
            seen.add(key)
    return torch.tensor(tasks, dtype=torch.float32).reshape(-1, 2, 2)


def new_policy():
    return nn.Sequential(nn.Linear(FEATURES, 48), nn.Tanh(), nn.Linear(48, 4))


def observe(quotes, seen, step):
    visible = quotes * seen.unsqueeze(-1)
    time_left = torch.full((len(quotes), 1), (HORIZON - step) / HORIZON)
    return torch.cat((visible.flatten(1) / 30, seen.float(), time_left), dim=1)


@torch.no_grad()
def rollout(policy, quotes, *, greedy=False, rule=False):
    """0/1 query a vendor. 2/3 submit that vendor and finish immediately."""
    n = len(quotes)
    rows = torch.arange(n)
    seen = torch.zeros(n, 2, dtype=torch.bool)
    alive = torch.ones(n, dtype=torch.bool)
    winner = quotes.sum(-1).argmin(-1)
    columns = {k: [] for k in ("obs", "action", "old_logp", "mask")}
    verified = torch.zeros(n, dtype=torch.bool)
    correct = torch.zeros_like(verified)
    submitted = torch.zeros_like(verified)
    queries = torch.zeros(n)
    for step in range(HORIZON):
        obs = observe(quotes, seen, step)
        if rule:
            # Demonstrator may choose a vendor only from the visible observations.
            visible_cost = obs[:, :4].reshape(-1, 2, 2).sum(-1)
            action = visible_cost.argmin(-1) + 2 if step == 2 else torch.full((n,), step)
            logp = torch.zeros(n)
        else:
            logits = torch.zeros(n, 4) if policy is None else policy(obs)
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if greedy else dist.sample()
            logp = dist.log_prob(action)
        columns["obs"].append(obs)
        columns["action"].append(action)
        columns["old_logp"].append(logp)
        columns["mask"].append(alive.clone())
        query = alive & (action < 2)
        commit = alive & (action >= 2)
        correct |= commit & (action - 2 == winner)
        verified |= commit & (action - 2 == winner) & seen.all(-1)
        submitted |= commit
        queries += query.float()
        seen[rows[query], action[query]] = True
        alive &= ~commit
    data = {k: torch.stack(v, dim=1) for k, v in columns.items()}
    data.update(verified=verified, correct=correct, submitted=submitted, queries=queries)
    return data


def supervised_fit(policy, demos):
    demo = rollout(None, demos, rule=True)
    opt = torch.optim.Adam(policy.parameters(), lr=0.012)
    for _ in range(80):
        logits = policy(demo["obs"][demo["mask"]])
        loss = nn.functional.cross_entropy(logits, demo["action"][demo["mask"]])
        opt.zero_grad()
        loss.backward()
        opt.step()


def rl_fit(policy, tasks, args, proxy):
    # Reference is the policy at the start of this RL run.
    reference = copy.deepcopy(policy).requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=0.003)
    log = []
    for step in range(args.steps):
        indices = torch.randint(len(tasks), (args.batch,))
        quotes = tasks[indices].repeat_interleave(args.group, dim=0)
        data = rollout(policy, quotes)
        rewards = data["submitted" if proxy else "verified"].float().view(-1, args.group)
        sd = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = ((rewards - rewards.mean(1, keepdim=True)) / (sd + 1e-6)).flatten()
        obs, act = data["obs"], data["action"]
        mask, old = data["mask"], data["old_logp"]
        ref_logp = reference(obs).log_softmax(-1)
        # Two updates on one frozen behavior batch make clipping meaningful.
        for _ in range(2):
            logp_all = policy(obs).log_softmax(-1)
            logp = logp_all.gather(-1, act.unsqueeze(-1)).squeeze(-1)
            ratio = (logp - old).exp()
            advantage = advantages.unsqueeze(1)
            surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
            kl = (logp_all.exp() * (logp_all - ref_logp)).sum(-1)
            # Sum real actions per trajectory, then average trajectories.
            # This is a teaching choice, distinct from per-token/per-length means.
            loss = ((-surrogate + args.beta * kl) * mask).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        log.append(dict(update=step + 1, train_reward=rewards.mean().item(),
                        verified=data["verified"].float().mean().item(),
                        tool_calls=int(data["queries"].sum().item()),
                        zero_group=(sd == 0).float().mean().item(),
                        ref_kl=kl.detach()[mask].mean().item(),
                        outside_clip=((ratio.detach()[mask] - 1).abs() > 0.2).float().mean().item()))
    return log


def metrics(data):
    return {k: round(data[k].float().mean().item(), 4)
            for k in ("verified", "correct", "submitted", "queries")}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def experiment(args):
    torch.set_num_threads(1)
    tasks = make_tasks("train", 1024)
    heldout = {split: make_tasks(split, 256) for split in ("dev", "test", "shift")}
    scores, curves, cases = [], [], []
    for seed in args.seeds:
        for method in args.methods:
            torch.manual_seed(seed)
            policy = None if method in ("rule", "random") else new_policy()
            if method.startswith("sft"):
                # All SFT arms for a given seed get identical demonstrations.
                demos = tasks[torch.randperm(len(tasks))[:args.demos]]
                supervised_fit(policy, demos)
            if method in ("grpo", "sft-grpo", "sft-proxy"):
                torch.manual_seed(seed + 10000)
                for point in rl_fit(policy, tasks, args, method == "sft-proxy"):
                    curves.append(dict(method=method, seed=seed, **point))
            for split, quotes in heldout.items():
                torch.manual_seed(seed + 20000)
                data = rollout(policy, quotes, greedy=policy is not None, rule=method == "rule")
                scores.append(dict(method=method, seed=seed, split=split, n=len(quotes), **metrics(data)))
                if args.cases:
                    for i, quote in enumerate(quotes.int().flatten(1).tolist()):
                        actions = data["action"][i, data["mask"][i]].tolist()
                        cases.append(dict(method=method, seed=seed, split=split, task_id=i,
                                          price_a=quote[0], freight_a=quote[1], price_b=quote[2],
                                          freight_b=quote[3], actions=" ".join(map(str, actions)),
                                          verified=int(data["verified"][i])))
    for row in scores:
        print("{method:10s} {seed:3d} {split:5s} verified={verified:.3f} "
              "correct={correct:.3f} submitted={submitted:.3f} queries={queries:.2f}".format(**row))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output / "scores.csv", scores)
        if curves:
            write_csv(args.output / "learning.csv", curves)
        if cases:
            write_csv(args.output / "cases.csv", cases)
        config = {**vars(args), "output": str(args.output), "torch": str(torch.__version__)}
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 19])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--demos", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", action="store_true", help="Include per-task traces in the output directory.")
    args = parser.parse_args()
    if args.group < 2 or min(args.batch, args.steps, args.demos) < 1 or args.demos > 1024 or args.beta < 0:
        parser.error("Require group >= 2, positive batch/steps, 1 <= demos <= 1024, beta >= 0.")
    if args.cases and args.output is None:
        parser.error("--cases requires --output.")
    experiment(args)
