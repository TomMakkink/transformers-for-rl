import math
import torch
import torch.nn as nn
import numpy as np


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        HS = self.hidden_size

        full_f_t_activations = np.zeros((self.hidden_size, seq_sz))

        f_t_activations = []
        i_t_activations = []
        o_t_activations = []
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS : HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2 : HS * 3]),
                torch.sigmoid(gates[:, HS * 3 :]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
            with torch.no_grad():
                full_f_t_activations[:, t] = f_t.detach().cpu().numpy().reshape((-1))[:]
                f_t_activations.append(torch.min(f_t).item())
                f_t_activations.append(torch.max(f_t).item())
                f_t_activations.append(torch.mean(f_t).item())
                f_t_activations.append(torch.std(f_t).item())

                i_t_activations.append(torch.min(i_t).item())
                i_t_activations.append(torch.max(i_t).item())
                i_t_activations.append(torch.mean(i_t).item())
                i_t_activations.append(torch.std(i_t).item())

                o_t_activations.append(torch.min(o_t).item())
                o_t_activations.append(torch.max(o_t).item())
                o_t_activations.append(torch.mean(o_t).item())
                o_t_activations.append(torch.std(o_t).item())

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        gate_activations = {
            "forget_gate": f_t_activations,
            "input_gate": i_t_activations,
            "output_gate": o_t_activations,
            "full_f_t_activations": full_f_t_activations,
        }
        return hidden_seq, (h_t, c_t), gate_activations
