# Taken from FastPitch
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from fs2.utils.model import sequence_mask
from fs2.models.components.nets.utils import SSSUnit

class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_inner, kernel_size[0], padding=(kernel_size[0] - 1) // 2 ),
            ReLUGTz(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size[1], padding=(kernel_size[1] - 1) // 2),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out).to(inp.dtype))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out).to(inp.dtype)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

        self.rope = RotaryEmbedding(d_head)

    def forward(self, inp, attn_mask=None):
        return self._forward(inp, attn_mask)

    def _forward(self, inp, attn_mask=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)


        q = self.rope.rotate_queries_or_keys(q)
        k = self.rope.rotate_queries_or_keys(k)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)

        output = output.to(attn_out.dtype)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout,
                 **kwargs):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, mask=None):
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2))
        output *= mask
        output = self.pos_ff(output)
        output *= mask
        return output


class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = padding_idx

        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_embed or d_model)
            torch.nn.init.normal_(self.word_emb.weight, 0.0, self.d_model**-0.5)
        else:
            self.word_emb = None

        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

    def forward(self, dec_inp, seq_lens=None, conditioning=0):
        if self.word_emb is None:
            inp = dec_inp
            mask = sequence_mask(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = sequence_mask(seq_lens).unsqueeze(2) 

        out = self.drop(inp + conditioning)

        for layer in self.layers:
            out = layer(out, mask=mask)

        # out = self.drop(out)
        return out, mask



class Encoder(nn.Module):
    """
    FFTransformer but with built in style conditioning
    """
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True,
                 n_embed=None, d_embed=None, padding_idx=0, pre_lnorm=False, style_dim=0):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.padding_idx = padding_idx

        if embed_input:
            self.word_emb = nn.Embedding(n_embed, d_embed or d_model)
            torch.nn.init.normal_(self.word_emb.weight, 0.0, self.d_model**-0.5)
        else:
            self.word_emb = None

        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()
        self.style_layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

            self.style_layers.append(
                SSSUnit(d_model, style_dim)
            )

    def forward(self, dec_inp, seq_lens=None, conditioning=0, style_vector=0):
        if self.word_emb is None:
            inp = dec_inp
            mask = sequence_mask(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = sequence_mask(seq_lens).unsqueeze(2) 

        out = self.drop(inp + conditioning)

        #for layer in self.layers:
        #    out = layer(out, mask=mask)


        for layer, style_layer in zip(self.layers, self.style_layers):
            out = layer(out, mask=mask) # encoder
            out = style_layer(out, style_vector) # injector
        # out = self.drop(out)
        return out, mask






# from https://github.com/ZDisket/FastSpeech2
class ReLUGTz(nn.Module):
    def __init__(self, alpha_init=0.5, beta_init=0.9, threshold_init=0.0, bias_init=0.0, shared_axes=None):
        super().__init__()
        self.dprelu = DPReLU(alpha_init, beta_init, threshold_init, bias_init, shared_axes)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = self.dprelu(x1) * x2
        return x

class DPReLU(nn.Module):
    """
    DPReLU: A dynamic ReLU variant:

    "There are four additional learnable parameters compared to the vanilla ReLU. alpha and beta are the slopes of the negative
    and positive parts in the function, respectively. Here, a negative or positive case is determined when comparing input
    x to the threshold. The threshold makes DPReLU shift on the x-axis in comparison to the original ReLU. The bias
    determines the alignment of the function with respect to the y-axis. These four parameters are all learnable and interact
    with each other during the training phase"

    https://link.springer.com/article/10.1007/s44196-023-00186-w

    By default, alpha and beta are 0.5 and 0.9, which yielded best results according to the paper. Threshold and bias = 0.

    Important: Please use He or their custom initialization!

    Converted from Tensorflow from https://github.com/KienMN/Activation-Experiments/tree/master
    """

    def __init__(self, alpha_init=0.5, beta_init=0.9, threshold_init=0.0, bias_init=0.0, shared_axes=None):
        super(DPReLU, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
        self.bias = nn.Parameter(torch.tensor(bias_init))

        self.shared_axes = shared_axes
        if self.shared_axes is not None and not isinstance(self.shared_axes, (list, tuple)):
            self.shared_axes = [self.shared_axes]

    def forward(self, inputs):
        neg = -self.alpha * torch.relu(-inputs + self.threshold)
        pos = self.beta * torch.relu(inputs - self.threshold)
        return pos + neg + self.bias

    def extra_repr(self):
        return f'alpha={self.alpha.item()}, beta={self.beta.item()}, threshold={self.threshold.item()}, bias={self.bias.item()}, shared_axes={self.shared_axes}'