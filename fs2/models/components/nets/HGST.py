# adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
# MIT License
#
# Copyright (c) 2018 MagicGirl Sakura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.



import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, n_mel_channels, conv_filters, gru_hidden):
        super().__init__()
        K = len(conv_filters)
        filters = [1] + conv_filters
        self.n_mel_channels = n_mel_channels

        self.convs = nn.ModuleList([
            nn.Conv2d(filters[i], filters[i+1], 3, stride=2, padding=1)
            for i in range(K)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(conv_filters[i]) for i in range(K)
        ])

        # Calculate output channels after K convs
        out_ch = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(conv_filters[-1] * out_ch, gru_hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(gru_hidden, gru_hidden))



    def forward(self, inputs, input_lengths=None):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            input_lengths = input_lengths.cpu().numpy().astype(int)            
            out = nn.utils.rnn.pack_padded_sequence(
                        out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)

        out = self.fc(out.squeeze(0))
        out = torch.tanh(out)
        return out

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class STL(nn.Module):
    '''
    inputs --- [N, token_embedding_size//2]
    '''
    def __init__(self, conv_filters, gru_hidden, token_embedding_size,
                 n_style_token, attn_head):
        super().__init__()
        self.embed = nn.Parameter(
            torch.FloatTensor(n_style_token, token_embedding_size // attn_head)
        )
        self.attention = MultiHeadAttention(gru_hidden, token_embedding_size//attn_head,
                                            token_embedding_size, attn_head)
        init.normal_(self.embed, mean=0, std=0.5)
        self.dropout = nn.Dropout(0.1)


        self.dropout = nn.Dropout(0.1)

    #def forward(self, inputs): # Old Version
    #    N = inputs.size(0)
    #    query = inputs.unsqueeze(1)
    #    keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
    #    style_embed = self.attention(query, keys)

    #    return style_embed


    def forward(self, inputs, r=None):
        # Handle 3D input [N, 1, dim] by squeezing
        if inputs.dim() == 3 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if r is not None and r.dim() == 3 and r.size(1) == 1:
            r = r.squeeze(1)

        N = inputs.size(0)

        if r is not None:
            query = (r - inputs).unsqueeze(1)
        else:
            query = inputs.unsqueeze(1)

        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size // num_heads]
        style_embed = self.attention(query, keys)
        style_embed = self.dropout(style_embed)
        if r is not None:
            return style_embed + inputs.unsqueeze(1)
        else:
            return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out


class HGST(nn.Module):
    def __init__(self,
                 n_mel_channels,         # e.g. 80
                 conv_filters,           # [32,32,64,64,128,128]
                 gru_hidden,             # 128
                 token_size,             # 128
                 n_style_token,          # 10
                 attn_head,              # 4
                 n_style_layers,         # 3
                 p_drop_style=0.1):
        
        super().__init__()
        
        self.encoder = ReferenceEncoder(n_mel_channels, conv_filters, gru_hidden)

        self.stls = nn.ModuleList([
            STL(conv_filters, gru_hidden, token_size, n_style_token, attn_head)
            for _ in range(n_style_layers)
        ])

        self.p_drop_style = p_drop_style

        self.fallback_embed = nn.Parameter(torch.randn(gru_hidden) * 0.02)
    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        

        if self.training:
            mask = torch.rand(enc_out.size(0), device=enc_out.device) < self.p_drop_style
            enc_out = torch.where(
                mask.unsqueeze(-1),
                self.fallback_embed.unsqueeze(0).expand_as(enc_out),
                enc_out
            )

        for i, layer in enumerate(self.stls):
            if i == 0:
               style_embed = layer(enc_out, r=None)
            else:
                style_embed = layer(style_embed.squeeze(1), r=enc_out)


        return style_embed
