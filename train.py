
import os
import pickle
import sys
import shutil

import yaml
import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from absl import flags, logging
from ml_collections import config_flags

import datasets
from tng_halo.classifier import BinaryNodeClassifier

logging.set_verbosity(logging.INFO)

def train(config: ml_collections.ConfigDict):
    # set up work directory
    logging.info("Starting training run {} at {}".format(config.name, config.workdir))
    workdir = os.path.join(config.workdir, config.name)

    checkpoint_path = None
    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
        elif config.get('checkpoint', None) is not None:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        else:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory.")

    # copy config file to workdir as yaml format
    os.makedirs(workdir, exist_ok=True)
    config_dict = ml_collections.ConfigDict.to_dict(config)
    with open(os.path.join(workdir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)

    # read in the dataset and prepare the data loader for training
    # train_loader, val_loader = datasets.prepare_loaders(
        # config.option1,
        # config.option2,
        # etc.
    # )
    train_loader, val_loader = None, None

    # create model
    model = BinaryNodeClassifier(
        input_size=config.model.input_size,
        hidden_sizes=config.model.hidden_sizes,
        embed_size=config.model.embed_size,
        graph_layer=config.model.graph_layer,
        graph_layer_args=config.model.graph_layer_args,
        activation_name=config.model.activation_name,
        layer_norm=config.model.layer_norm,
        norm_first=config.model.norm_first,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler
    )

    # create the trainer object
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss', patience=config.training.patience,
            mode='min', verbose=True),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{step}-{val_loss:.4f}", monitor='val_loss',
            save_top_k=config.training.save_top_k, mode='min',
            save_weights_only=False),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_steps=config.training.num_steps,
        accelerator=config.training.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=config.get("enable_progress_bar", True),
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(config.training_seed)   # set random seed
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=checkpoint_path
    )

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    train(config=FLAGS.config)
