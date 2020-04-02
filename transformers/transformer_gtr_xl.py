import torch 
import torch.nn as nn 
from transformers.positional_encoding_layer import PositionalEncoding
from transformers.attention_layer import GTrXLBlock

Tensor = torch.Tensor

# TODO: Incorporate memory and relative positioning 
class GTrXL(nn.Module):
    """
    GTrXL Transformer model from the "Stabilizing Transformers for Reinforcement Learning" paper: 
    https://arxiv.org/abs/1910.06764. 
    
    The GTrXL modifies the Transformer-XL architecture by chaning 
    the position of the layer normalisation layer, and add additional Gated Recurrent Unit layer. 
    These modifications were demonstrated to improve the stability and learning speed of the original
    Transformer and Transformer XL in a variety of partially observable RL domains. 
    """
    def __init__(
        self, 
        d_model:int, 
        ninp:int, 
        output_dim:int,
        num_layers:int, 
        num_heads:int, 
        dim_mlp:int,    
        dropout:float=0.0,
    ):
        """
        Args: 
            d_model: number of expected features in the input. 
            ninp: number of inputs. 
            output_dim: output dimension of the model.  
            num_layers: number of submodules in the transformer. 
            num_heads: number of attention heads.  
            dim_mlp: inner dimension of multilayer perceptron. 
            dropout: dropout. Default: 0.0. 
        """
        super(GTrXL, self).__init__()
        self.positional_encoding_layer = PositionalEncoding(encoding_type="absolute", d_model=d_model)
        self.dropout = nn.Dropout(dropout)

        self.GTrXLs = [
            GTrXLBlock(
                num_heads=num_heads,
                d_model=d_model,
                dim_mlp=dim_mlp,
                use_scale=True,
                dropout=dropout, 
            )
            for k in range(num_layers)
        ]

        self.output_layer = nn.Linear(ninp * d_model, output_dim, bias=False)
        # self.output_layer = nn.Sequential(
        #     nn.Linear(d_model, dim_mlp, bias=False), nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_mlp, output_dim, bias=False)
        # )


    def forward(self, inputs:Tensor):
        """
        Args: 
            inputs: input tensor, of shape: [source_seq_len, batch_size, features]

        Returns: 
            GTrXL output of shape [output_dim]
        """
        x = self.positional_encoding_layer(inputs)
        for layer in self.GTrXLs:
            x = layer(x)
        x = torch.flatten(x)
        output = self.output_layer(x)
        return output