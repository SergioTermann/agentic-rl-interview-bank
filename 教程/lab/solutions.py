import torch


def action_logps(logits, ids, assistant, valid):
    targets = ids[:, 1:]
    log_dist = logits[:, :-1].float().log_softmax(-1)
    chosen = log_dist.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return chosen, assistant[:, 1:] & valid[:, 1:] & valid[:, :-1]


def gae(reward, value, next_value, bootstrap, carry, gamma, lam):
    with torch.no_grad():
        delta = reward + gamma * bootstrap * next_value - value
        out = torch.zeros_like(delta)
        running = torch.zeros_like(delta[:, 0])
        for t in reversed(range(delta.shape[1])):
            running = delta[:, t] + gamma * lam * carry[:, t] * running
            out[:, t] = running
    return out


def group_adv(rewards):
    rewards = rewards.detach().float()
    centered = rewards - rewards.mean(-1, keepdim=True)
    return centered / (rewards.std(-1, keepdim=True, unbiased=False) + 1e-6)


def clip_loss(new, old, adv, mask, epsilon=0.2):
    if not mask.any():
        raise ValueError("Empty action batch")
    ratio = (new[mask].float() - old.detach()[mask].float()).exp()
    a = adv.detach()[mask]
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - epsilon, 1 + epsilon) * a)
    return -surrogate.mean()
