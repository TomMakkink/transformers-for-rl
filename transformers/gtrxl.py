import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


"""
GRU gating layer used in Stabilizing transformers in RL.
Note that all variable names follow the notation from section: "Gated-Recurrent-Unit-type gating" 
in https://arxiv.org/pdf/1910.06764.pdf
"""


class GRUGate(nn.Module):
    def __init__(self, d_model):
        # d_model is dimension of embedding for each token as input to layer (want to maintain this in the gate)
        super(GRUGate, self).__init__()

        # TODO: DEBUG Make sure intitialize bias of linear_w_z to -3
        self.linear_w_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_z = nn.Linear(
            d_model, d_model
        )  ### Giving bias to this layer (will count as b_g so can just initialize negative)
        self.linear_u_z = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(
                -2
            )  # Manually setting this bias to allow starting with markov process
            # Note -2 is the setting used in the paper stable transformers

    def forward(self, x, y):
        ### Here x,y follow from notation in paper
        # TODO: DEBUG MAKE SURE THIS IS APPLIED ON PROPER AXIS
        z = torch.sigmoid(
            self.linear_w_z(y) + self.linear_u_z(x)
        )  # MAKE SURE THIS IS APPLIED ON PROPER AXIS
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(
            self.linear_w_g(y) + self.linear_u_g(r * x)
        )  # Note elementwise multiplication of r and x
        return (1.0 - z) * x + z * h_hat


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    # QUESTION: WHAT IS shape of inp to make this work (elementwise fnn and have batch dim)
    def forward(self, inp):

        ##### positionwise feed-forward (this is what's used in original transformer)
        core_out = self.CoreNet(inp)

        return core_out


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        use_gate,
        use_stable_version,
        **kwargs,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.use_gate = use_gate
        self.use_stable_version = use_stable_version

        if self.use_gate:
            self.gate_mha = GRUGate(d_model)
            self.gate_mlp = GRUGate(d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs
        )
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward_orig(
        self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None
    ):

        output = self.dec_attn(
            dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        output = self.layer_norm1(dec_inp + output)
        output2 = self.pos_ff(output)
        output2 = self.layer_norm2(output + output2)
        return output2

    def forward_stable(
        self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None
    ):

        # Layer norm will be applied at start of MHA module on both dec_inp2 and mems
        # dec_inp2 = self.layer_norm1(dec_inp)
        # First Layer norm will be applied within dec_attn

        dec_inp2 = self.dec_attn(
            dec_inp,
            r,
            r_w_bias,
            r_r_bias,
            attn_mask=dec_attn_mask,
            mems=mems,
            use_stable_version=self.use_stable_version,
        )

        # NOTE: In stable transformer they apply Relu before the layernorm/gate (in appendix C.3)
        if self.use_gate:
            dec_inp2 = self.gate_mha(dec_inp, F.relu(dec_inp2))
        else:
            dec_inp2 = dec_inp + F.relu(dec_inp2)

        dec_inp3 = self.layer_norm2(dec_inp2)

        dec_inp3 = self.pos_ff(dec_inp3)

        if self.use_gate:
            dec_inp3 = self.gate_mlp(dec_inp2, F.relu(dec_inp3))
        else:
            dec_inp3 = F.relu(dec_inp3) + dec_inp2

        return dec_inp3

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        if self.use_stable_version:
            return self.forward_stable(
                dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems
            )

        return self.forward_orig(dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems)


class RelMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        tgt_len=None,
        ext_len=None,
        mem_len=None,
        pre_lnorm=False,
    ):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # Get query, key and value for each token (NOTE SOME Inefficiency since
        # don't need query for any of the memory. Parallelization must make up for it
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        # UserWarning: masked_fill_ received a mask with dtype torch.uint8,
        # this behavior is now deprecated,please use a mask with dtype torch.bool instead.
        # changed .byte() to .bool()
        # mask = torch.ones((h, w)).byte()
        mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros(
                (x.size(0), qlen - 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]).view(
            qlen, klen, x.size(2), x.size(3)
        )

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(
        self,
        w,
        r,
        r_w_bias,
        r_r_bias,
        attn_mask=None,
        mems=None,
        use_stable_version=False,
    ):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        # if using stable version, then want layernorm of memory as well before MHA
        if mems is not None:
            cat = torch.cat([mems, w], 0)

            w_heads = (
                self.qkv_net(cat)
                if not use_stable_version
                else self.qkv_net(self.layer_norm(cat))
            )
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = (
                self.qkv_net(w)
                if not use_stable_version
                else self.qkv_net(self.layer_norm(w))
            )
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(
            qlen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(
            rlen, self.n_head, self.d_head
        )  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum(
            "ibnd,jbnd->ijbn", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum(
            "ibnd,jnd->ijbn", (rr_head_q, r_head_k)
        )  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[None, :, :, None], -float("inf"))
                    .type_as(attn_score)
                )
            elif attn_mask.dim() == 3:  # THIS IS WHAT IS Usually executed
                # print('Attentionscore shape: ',attn_score.shape)
                # print('MASK SHAPE: ', attn_mask[:,:,:,None].shape)
                # print('MASK EL 1: ', attn_mask[:,:,0])
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[:, :, :, None], -float("inf"))
                    .type_as(attn_score)
                )

        # print('ATTENTION SCORE: ', attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection + layer normalization
        # output = self.layer_norm(w + attn_out)

        return attn_out


# TODO : DEBUG, sanity check the memtransformerLM implementation with the one in the Stabilizing paper
class MemTransformerLM(nn.Module):
    def __init__(
        self,
        n_token,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        tie_weight=True,
        d_embed=None,
        div_val=1,
        tgt_len=None,
        ext_len=0,
        mem_len=1,
        cutoffs=[],
        adapt_inp=False,
        same_length=False,
        clamp_len=-1,
        use_gate=True,
        use_stable_version=True,
    ):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token  # TODO : Check this is not being used anywhere

        self.d_embed = d_model
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        # self.state_emb = State_Embedder()

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        # self.max_klen = tgt_len + ext_len + mem_len

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    dropout,
                    use_stable_version=use_stable_version,
                    use_gate=use_gate,
                    tgt_len=tgt_len,
                    ext_len=ext_len,
                    mem_len=mem_len,
                    dropatt=dropatt,
                )
            )

        # To do: Look into sample softmax and adaptive softmax for future, not relevant here though
        # are useful when need fast softmax over many classes

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def init_gru_bias(self):
        for l in self.layers:
            l.gate_mha.init_bias()
            l.gate_mlp.init_bias()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    # NOTE: qlen looks to be number of characters in one example
    #      mlen is memory size
    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(
                0, qlen - 0 - self.ext_len
            )  # ext_len looks to usually be 0 (in their experiments anyways

            # TODO: I have changed beg_idx to 0 since want to use all memory, may want to change
            #       this once move to larger environments (THIS HAS NOW BEEN CHANGED)

            # HERE IS THE PROBLEM.
            # print('hids shape: ', hids[0].shape)

            beg_idx = max(0, end_idx - self.mem_len)  # if hids[0].shape[0] > 1 else 0
            # print('BEG IND: ', beg_idx)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    # def _forward(self, dec_inp, obs_emb, mems=None):
    # TODO : We dropped dec_input since the first 2 dims of obs_emb should be the same as
    # that of dec_input, which is unrolled length = query length and batch_size
    # we saw this from             core_input = core_input.view(T, B, -1) line 668 in monobeast_test.py
    def _forward(self, obs_emb, mems=None):

        qlen, bsz, _ = obs_emb.size()  # qlen is number of characters in input ex

        if mems is not None:
            mlen = mems[0].size(0)
            # print('HERE: mlen: {}, len mems: {}, mems[0] shape: {}'.format(mlen, len(mems),mems[0].shape))
        else:
            mlen = 0
        # mlen = mems[0].size(0) if mems is not None else 0

        klen = mlen + qlen

        # create the mask taking in consideration the mlen as well. All memory should be attended by the first query
        dec_attn_mask = torch.triu(
            obs_emb.new_ones(qlen, klen), diagonal=1 + mlen
        ).bool()[:, :, None]

        hids = []
        pos_seq = torch.arange(
            klen - 1, -1, -1.0, device=obs_emb.device, dtype=obs_emb.dtype
        )
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(obs_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        # SEEMS THAT THEY store memory per layer which makes sense to attend to (for ex if at first layer, if we were
        # applying attention to memory and this new data, this would give us the same result.
        for i, layer in enumerate(self.layers):
            # print('HIDDEN iter: {}, output: {}'.format(i, core_out[-1,0, :10]))

            # TODO : The memory should be the same hidden layer's state of the previous T timesteps
            mems_i = None if mems is None else mems[i]
            # print('from txl483 shapes : ', core_out.shape, pos_emb.shape, self.r_w_bias.shape, self.r_r_bias.shape, dec_attn_mask.shape, mems_i.shape)
            core_out = layer(
                core_out,
                pos_emb,
                self.r_w_bias,
                self.r_r_bias,
                dec_attn_mask=dec_attn_mask,
                mems=mems_i,
            )
            hids.append(core_out)

        core_out = self.drop(core_out)
        # print('before update mems hids shape: {}, mems shape {}'.format(hids[0].shape,mems[0].shape if mems else None))
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, mems):
        if not mems:
            # print("INITIALIZED MEMS")
            mems = self.init_mems()

        hidden, new_mems = self._forward(data, mems=mems)

        return hidden, new_mems
