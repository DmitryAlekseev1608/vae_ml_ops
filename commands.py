import os
from os.path import dirname, split

import fire
import mlflow
import torch
from git import Repo
from hydra import compose, initialize
from mlflow import MlflowClient

from vae_ml_ops.infer import infer as start_infer
from vae_ml_ops.train import train as start_train


def _get_git_commit_id():

    """Return the ID of the git HEAD commit."""

    path = split(dirname(__file__))[0] + "/vae_ml_ops"
    commit_id = Repo(path).head.object.hexsha

    return commit_id[:8]


def print_auto_logged_info(r):
    "Print logs"
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


def train():
    """Point entry for train"""

    os.system('dvc pull --remote gd_vae --force')

    mlflow.set_tracking_uri(uri=cfg.ml_ops.mlflow_server_test)
    mlflow.set_experiment("train VAE")

    with mlflow.start_run() as run:

        commit_id = _get_git_commit_id()
        params = {
            "Epoch": cfg.train.n_epochs,
            "Image_size": cfg.model.size_img,
            "Features": cfg.model.features,
            "commit_id": commit_id,
        }

        mlflow.log_params(params)

        autoencoder = start_train(cfg)

        mlflow.set_tag("Training Info", "VAE model for fetch_dataset")

        mlflow.pytorch.log_model(autoencoder, "model")
        scripted_pytorch_model = torch.jit.script(autoencoder)
        mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

        for artifact_path in ["model/data", "scripted_model/data"]:
            [
                f.path
                for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)
            ]

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    os.system('dvc add models/autoencoder')
    os.system('dvc push --remote gd_vae --with-deps models/autoencoder')


def infer():
    """Point entry for infer"""

    os.system('dvc pull --remote gd_vae --force --with-deps models/autoencoder')

    mlflow.set_tracking_uri(uri=cfg.ml_ops.mlflow_server_test)
    mlflow.set_experiment("infer VAE")

    with mlflow.start_run():

        commit_id = _get_git_commit_id()
        params = {
            "Epoch": cfg.train.n_epochs,
            "Image_size": cfg.model.size_img,
            "Features": cfg.model.features,
            "commit_id": commit_id,
        }
        mlflow.log_params(params)
        start_infer(cfg)
        mlflow.set_tag("Infering Info", "VAE model for fetch_dataset")


if __name__ == '__main__':

    initialize(version_base=None, config_path="configs", job_name="app")
    cfg = compose(config_name="config")
    fire.Fire()
