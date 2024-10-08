
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GNNBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layer_name: str,
        layer_args: Optional[Dict[str, Any]] = None,
        activation_fn: callable = nn.ReLU(),
        layer_norm: bool = False,
        norm_first: bool = False
    ) -> None:
        """
        A block of a graph neural network.

        Parameters
        ----------
        input_size : int
            The size of the input features.
        output_size : int
            The size of the output features.
        layer_name : str
            The name of the graph layer.
        layer_args : dict, optional
            The keyword arguments for the graph layer.
        activation_fn : callable, optional
            The activation function.
        layer_norm : bool, optional
            Whether to use layer`` normalization.
        norm_first : bool, opti`onal
            Whether to apply normalization before the activation function.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_name = layer_name
        self.layer_args = layer_args or {}
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm
        self.norm_first = norm_first
        self.has_edge_attr = False
        self.has_edge_weight = False
        self.graph_layer = None
        self.norm = None

        self._setup_model()

    def _setup_model(self) -> None:
        if self.layer_name == "ChebConv":
            self.has_edge_attr = False
            self.has_edge_weight = True
            self.graph_layer =  gnn.ChebConv(
                self.input_size, self.output_size, **self.layer_args)
        elif self.layer_name == "GCNConv":
            self.has_edge_attr = False
            self.has_edge_weight = True
            self.graph_layer =  gnn.GCNConv(
                self.input_size, self.output_size, **self.layer_args)
        elif self.layer_name == "GATConv":
            self.has_edge_attr = True
            self.has_edge_weight = False
            self.graph_layer =  gnn.GATConv(
                self.input_size, self.output_size, **self.layer_args)
        else:
            raise ValueError(f"Unknown graph layer: {layer_name}")

        if self.layer_norm:
            self.norm = gnn.norm.LayerNorm(self.output_size)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        # apply graph layer
        if self.has_edge_attr:
            x = self.graph_layer(x, edge_index, edge_attr)
        elif self.has_edge_weight:
            x = self.graph_layer(x, edge_index, edge_weight)
        else:
            x = self.graph_layer(x, edge_index)

        # apply layer norm and activation
        if self.norm_first and self.norm is not None:
            x = self.norm(x)
            x = self.activation_fn(x)
        elif self.norm is not None:
            x = self.activation_fn(x)
            x = self.norm(x)
        else:
            x = self.activation_fn(x)
        return x


class MLPBatchNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [512],
        activation_fn: callable = nn.ReLU(),
        batch_norm: bool = False,
        dropout: float = 0.0
    ) -> None:
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        output_size : int
            The number of classes
        hidden_sizes : list of int, optional
            The sizes of the hidden layers. Default: [512]
        activation_fn : callable, optional
            The activation function to use. Default: nn.ReLU()
        batch_norm: bool, optional
            Whether to use batch normalization. Default: False
        dropout: float, optional
            The dropout rate. Default: 0.0
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn

        # Create a list of all layer sizes: input, hidden, and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(activation_fn)
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(out_dim))
            self.layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
