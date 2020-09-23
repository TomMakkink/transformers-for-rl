import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()
        self.writer = SummaryWriter()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def log_parameters(self, step):
        
        for i, weight in enumerate(self.parameters()):
            self.writer.add_histogram(f"lstm/{i}", weight.data, global_step=step)

    def forward(self, x,
                init_states=None, step=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        self.writer.add_histogram(f"lstm/h_t", h_t, global_step=step)
        self.writer.add_histogram(f"lstm/c_t", c_t, global_step=step)
        self.log_parameters(step)

        HS = self.hidden_size

        f_t_activations = []
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
            with torch.no_grad():
                f_t_activations.append(
                    [torch.min(f_t).item(),
                     torch.max(f_t).item(),
                     torch.mean(f_t).item(),
                     torch.std(f_t).item()])
                # print(
                #     f"{t}) i_t {torch.mean(i_t)}, f_t {torch.mean(f_t)}, g_t {torch.mean(g_t)}, o_t {torch.mean(o_t)}")
        # print('-' * 20)
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t), f_t_activations
