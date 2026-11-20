"""Copy this file to mine.py in this directory, then implement each function."""

import torch


def action_logps(logits, ids, assistant, valid):
    """[B,L,V], [B,L], two [B,L] bool masks -> chosen logps, action mask [B,L-1]."""
    raise NotImplementedError


def gae(reward, value, next_value, bootstrap, carry, gamma, lam):
    """Five [B,T] tensors. bootstrap and carry are separate 0/1 gates."""
    raise NotImplementedError


def group_adv(rewards):
    """[B,G] -> same shape, independent per task, population std, epsilon 1e-6."""
    raise NotImplementedError


def clip_loss(new, old, adv, mask, epsilon=0.2):
    """[B,T] logps, advantage, bool mask. Negative clipped objective per real action."""
    raise NotImplementedError
