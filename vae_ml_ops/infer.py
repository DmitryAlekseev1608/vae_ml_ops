import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from .model import LinearVAE


def infer(cfg):

    """
    Функция реализующая работу обученной модели
    """

    print(OmegaConf.to_yaml(cfg.model))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    autoencoder = LinearVAE(cfg).to(device)
    autoencoder.load_state_dict(torch.load("models/autoencoder"))
    autoencoder.eval()

    z = np.array([np.random.normal(0, 1, cfg.model.features) for i in range(10)])
    output = autoencoder.sample(torch.FloatTensor(z).to(device))

    plt.figure(figsize=(18, 18))
    for i in range(output.shape[0]):
        plt.subplot(output.shape[0] // 2, 2, i + 1)
        generated = output[i].cpu().detach().numpy()
        plt.imsave(f"result/result_{i}.jpg", generated)
