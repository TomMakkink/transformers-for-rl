import torch
import torch.nn as nn
from configs.transformer_config import transformer_config
from configs.lstm_config import lstm_config
from configs.experiment_config import experiment_config

from transformers.transformer import (
    TransformerModel,
    MemoryTransformerModel,
    get_transformer_submodule,
)

from transformers.gtrxl import MemTransformerLM


def init_weight(weight):
    # if args.init == 'uniform':
    #    nn.init.uniform_(weight, -args.init_range, args.init_range)
    # elif args.init == 'normal':
    nn.init.normal_(weight, 0.0, 0.02)  # args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("AdaptiveEmbedding") != -1:
        if hasattr(m, "emb_projs"):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, 0.01)  # args.proj_init_std)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            init_weight(m.weight)
    elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
        if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, "out_projs"):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, 0.01)  # args.proj_init_std)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, 0.02)  # args.init_std)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("TransformerLM") != -1:
        print("FOUND TRNASFORMER LM")
        if hasattr(m, "r_emb"):
            init_weight(m.r_emb)
        if hasattr(m, "r_w_bias"):
            init_weight(m.r_w_bias)
        if hasattr(m, "r_r_bias"):
            init_weight(m.r_r_bias)
        if hasattr(m, "r_bias"):
            init_bias(m.r_bias)


class Memory(nn.Module):
    """
    Memory wrapper that is either an LSTM or a Transformer. 
    """

    def __init__(self, memory_type, input_dim, output_dim):
        super(Memory, self).__init__()
        self.memory = None
        self.memory_type = None

        if memory_type is not None:
            self.memory_type = memory_type.lower()
            print(f"Using {self.memory_type}...")
            if self.memory_type == "lstm":
                self.memory = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=lstm_config["hidden_dim"],
                    num_layers=lstm_config["num_layers"],
                )
                self.hidden = (
                    torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
                        experiment_config["device"]
                    ),
                    torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
                        experiment_config["device"]
                    ),
                )
            elif self.memory_type == "stable":
                self.memory = MemTransformerLM(
                    n_token=None,
                    n_layer=transformer_config["num_layers"],
                    n_head=transformer_config["num_heads"],
                    d_head=input_dim // transformer_config["num_heads"],
                    d_model=input_dim,
                    d_inner=transformer_config["dim_mlp"],
                    dropout=0.1,
                    dropatt=0.0,
                    mem_len=transformer_config["mem_len"],
                    use_stable_version=True,
                    use_gate=True,
                )
                self.memory.apply(weights_init)
                self.memory.init_gru_bias()
                self.mem = None

            elif self.memory_type in ["vanilla", "rezero", "linformer", "mha", "lmha"]:
                submodule = get_transformer_submodule(self.memory_type)
                self.memory = TransformerModel(input_dim, output_dim, submodule)
            elif self.memory_type in ["gtrxl", "xl", "rmha", "gmha"]:
                submodule = get_transformer_submodule(self.memory_type)
                self.mem = None
                self.memory = MemoryTransformerModel(input_dim, output_dim, submodule)

    def forward(self, x):
        """
        x: shape [seq_len, batch_size, feature_dim]
        """
        if type(self.memory) is nn.LSTM:
            batch_size = x.shape[1]
            if batch_size > 1:
                print("Batch Size above 1")
                x, self.hidden = self.memory(x)
            else:
                x, self.hidden = self.memory(x, self.hidden)
        elif self.memory_type == "stable":
            output, self.mem = self.memory(x, self.mem)
        elif type(self.memory) == MemoryTransformerModel:
            output, self.mem = self.memory(x, self.mem)
        elif type(self.memory) == TransformerModel:
            x = self.memory(x)
        return x

    def reset(self):
        if self.memory_type == "lstm":
            self.hidden = (
                torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
                    experiment_config["device"]
                ),
                torch.zeros(1, 1, lstm_config["hidden_dim"]).to(
                    experiment_config["device"]
                ),
            )
        elif self.memory_type == "stable":
            self.mem = None
        elif type(self.memory) == MemoryTransformerModel:
            self.mem = None
