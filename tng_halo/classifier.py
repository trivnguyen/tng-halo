
from typing import Any, Dict, List
import torch
import torch.nn as nn
import pytorch_lightning as pl

from tng_halo.models import GNNBlock
from tng_halo import training_utils

class BinaryNodeClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        embed_size: int = None,
        graph_layer: str = "ChebConv",
        graph_layer_args: Dict[str, Any] = None,
        activation_name: callable = nn.ReLU(),
        layer_norm: bool = False,
        norm_first: bool = False,
        optimizer_args: Dict[str, Any] = None,
        scheduler_args: Dict[str, Any] = None
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.embed_size = embed_size
        self.graph_layer = graph_layer
        self.graph_layer_args = graph_layer_args or {}
        self.activation_name = activation_name
        self.layer_norm = layer_norm
        self.norm_first = norm_first
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.save_hyperparameters()
        self._setup_model()

    def _setup_model(self) -> None:
        if self.embed_size:
            self.embed_layer = nn.Linear(self.input_size, self.embed_size)
            input_size = self.embed_size
        else:
            self.embed_layer = None
            input_size = self.input_size

        layer_sizes = [input_size] + self.hidden_sizes
        activation_fn = training_utils.get_activation(self.activation_name)

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = GNNBlock(
                layer_sizes[i-1],
                layer_sizes[i],
                layer_name=self.graph_layer,
                layer_args=self.graph_layer_args,
                layer_norm=self.layer_norm,
                norm_first=self.norm_first,
                activation_fn=activation_fn
            )
            self.layers.append(layer)
        self.output_layer = GNNBlock(
            layer_sizes[-1], 1,
            layer_name=self.graph_layer,
            layer_args=self.graph_layer_args,
            layer_norm=self.layer_norm,
            norm_first=self.norm_first,
            activation_fn=nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:

        if self.embed_layer:
            x = self.embed_layer(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, edge_weight)
        x = self.output_layer(x, edge_index, edge_attr, edge_weight)
        return x

    def training_step(self, batch, batch_idx):
        yhat = self.forward(
            batch.x, batch.edge_index, batch.edge_attr, batch.edge_weight)
        loss = F.binary_cross_entropy_with_logits(yhat, batch.y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation(self, batch, batch_idx):
        yhat = self.forward(
            batch.x, batch.edge_index, batch.edge_attr, batch.edge_weight)
        loss = F.binary_cross_entropy_with_logits(yhat, batch.y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return training_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
