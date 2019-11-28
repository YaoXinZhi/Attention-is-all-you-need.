#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 26/11/2019 9:54 
@Author: XinZhi Yao 
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.
    """
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale=None, attn_mask=None):
        """
        Feed-Forward
        :param q: Queries Tensor, [batch_size, length_q, dim_q]
        :param k: Keys Tensor, [batch_size, length_k, dim_k]
        :param v: Valuse Tensor, [batch_size, length_v, dim_v], generally it's k.
        :param scale: scaling factor, a float scale.
        :param attn_mask: Masking Tensor, [batch_size, length_q, length_k]
        :return: context tensor and attention tensor.
        attention: [batch_size, length_q, length_k]
        context: [batch_size, length_q, dim_v]
        """
        # torch.bmm() batch matrix multiplication
        # (b, x1, y) * (b, y, x2) = (b, x1, x2)
        # Calculate similarity
        attention = torch.bmm(q, k.transpose(1,2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        # calculation of context
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q, K, V) = Concat(head_1,...,head_h)W^o
    where, head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.all_heads_dim = self.dim_per_head * num_heads

        self.linear_k = nn.Linear(model_dim, self.all_heads_dim)
        self.linear_v = nn.Linear(model_dim, self.all_heads_dim)
        self.linear_q = nn.Linear(model_dim, self.all_heads_dim)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(self.all_heads_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        # after multi-head attention need a layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # Residual connection
        residual = query

        batch_size = key.size(0)

        # linear projection [b, l, all_head_dim]
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            # [batch_size, *, *] --> [batch_size * num_heads, *, *]
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        # scaled dot prodct attention (scaling factor 1/sqrt(d_k))
        # For each head
        scale = (key.size(-1) // self.num_heads) ** -0.5
        # for dot_product_attention:
        #   input: [batch_size, length_q/k/v, dim_q/k/v]
        #   attention: [batch_size, length_q, length_k]
        #   context: [batch_size, length_q, dim_v]
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask
        )

        # concat heads
        context = context.view(batch_size, -1, self.all_heads_dim)

        # final linear projection
        final_values = self.linear_final(context)
        final_values = self.dropout(final_values)
        # add residual and norm layer
        final_values = self.layer_norm(residual + final_values)
        return final_values, attention


def padding_mask(seq_key: torch.Tensor, seq_query: torch.Tensor):
    """
    padding mask
    :param seq_key: sequence of key, [batch_size, length_k]
    :param seq_query: sequence of query, [batch_size, length_q]
    :return: padding mask squence, [batch_size, length_q, length_k]
    """
    length_query = seq_query.size(1)
    # '<pad>' == 0
    pad_mask = seq_key.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, length_query, -1)
    return pad_mask

def sequence_mask(seq: torch.Tensor):
    """
    sequence mask
    :param seq: sequence needed to mask. [batch_size, length_seq]
    :return: mask matrix. [batch_size, length_seq, length_seq]
    """
    batch_size, seq_len = seq.size()
    # Returns a tensor containing the upper triangular part of
    # the input matrix (2D tensor) and the rest is set to 0.
    seq_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                      diagonal=1)
    seq_mask = seq_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return seq_mask

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, max_seq_len: int):
        """
        init PositionalEncoding
        :param dim_model: A scale, dimension of model, in paper is 512.
        :param max_seq_len: A scale, the max length of sequence.
        :param pisition_embedding: [vocab_size+1, dim_model]
        """
        super(PositionalEncoding, self).__init__()
        # PE matrix
        position_encoding = np.array([
            [pos/np.power(10000, 2.0*(j//2)/dim_model)for j in range(dim_model)]\
            for pos in range(max_seq_len)])
        # Use sin for even columns and cos for odd columns
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # In the first row of the PE matrix, add a vector of all zeros to
        # represent the positional encoding of '<PAD>'.
        pad_row = torch.zeros([1, dim_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        # '<PAD>' --> max_seq_len + 1
        self.position_embedding = nn.Embedding(max_seq_len + 1, dim_model)
        self.position_embedding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, input_length: torch.Tensor):
        """
        Feed forward of network.
        :param input_length: A tensor, [batch_size, 1], each dim is the length of sequence.
        :return: position_embedding [batch_size, max_seq_len+1, dim_model]
        """
        max_len = int(torch.max(input_length))
        tensor = torch.cuda.LongTensor if input_length.is_cuda else torch.LongTensor
        # here range starts from 1 because it is to acid '<PAD>'
        input_pos = tensor(
            [ list(range(1, length + 1)) + [0] * (max_len - int(length)) for length in input_length]
        )
        return self.position_embedding(input_pos)

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

# Transformer
class EncoderLayer(nn.Module):
    """
    One Encoder Layer for Transformer
    """
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention

class Encoder(nn.Module):
    """
    Encoder of Transformer
    """
    def __init__(self, vocab_size, max_seq_len, num_layers=5, model_dim=512,
                 num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads,
                                                          ffn_dim, dropout)\
                                            for _ in range(num_layers)])
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)
        padding_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, padding_mask)
            attentions.append(attention)
        return output, attentions

class DecoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, padding_seq_mask=None, context_padding_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
            dec_inputs, dec_inputs, dec_inputs, padding_seq_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
            enc_outputs, enc_outputs, dec_output, context_padding_mask)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention

class Decoder(nn.Module):

    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512,
                 num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout)\
             for _ in range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size+1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        dec_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, dec_attn_mask, context_attn_mask
            )
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)
        return output, self_attentions, context_attentions

class Transformer(nn.Module):

    def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len,
                 num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)
        output, enc_self_attn = self.encoder(src_seq, src_len)

        output, dec_self_attn, context_attn = self.decoder(
            tgt_seq, tgt_len, output, context_attn_mask
        )
        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, context_attn
