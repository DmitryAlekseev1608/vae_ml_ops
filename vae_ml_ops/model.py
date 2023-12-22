import torch
import torch.nn as nn


# define a simple linear VAE


class LinearVAE(nn.Module):
    """
    Объявление класса модели
    """

    def __init__(self, cfg):
        super(LinearVAE, self).__init__()

        self.features = cfg.model.features
        self.size = cfg.model.size_img

        self.flatten = nn.Flatten()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=3 * self.size**2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.features * 2),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=3 * self.size**2),
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        """Прямой проход по нейронной сети"""
        # encoding
        x = self.flatten(x).float()
        x = self.encoder(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = self.decoder(z)
        reconstruction = torch.sigmoid(x)
        return reconstruction, mu, log_var

    def sample(self, z):
        """Генерирование примеров"""
        generated = self.decoder(z)
        generated = torch.sigmoid(generated)
        generated = generated.view(-1, self.size, self.size, 3)
        return generated

    def get_latent_vector(self, x):
        """Получение изображений из латентного пространства"""
        x = self.flatten(x).float()
        x = self.encoder(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return z
