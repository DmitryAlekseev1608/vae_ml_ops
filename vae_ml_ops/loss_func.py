import torch
import torch.nn as nn


def KL_divergence(mu, logsigma):
    """
    часть функции потерь, которая отвечает за "близость" латентных
    представлений разных людей
    """
    loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return loss


def log_likelihood(x, reconstruction):
    """
    часть функции потерь, которая отвечает за качество реконструкции
    """
    loss = nn.BCELoss(reduction='sum')
    return loss(reconstruction, x)


def loss_vae(x, mu, logsigma, reconstruction):
    """Суммарная функция потерь"""
    return KL_divergence(mu, logsigma) + log_likelihood(x, reconstruction)
