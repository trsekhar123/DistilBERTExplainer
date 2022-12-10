""" PyTorch DistilBERT model
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
    and in part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import copy
import sys
from io import open

import itertools
import numpy as np

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from transformers import DistilBertConfig
# from .file_utils import add_start_docstrings

from explain_lib.utils import *

import logging
logger = logging.getLogger(__name__)




DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'distilbert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-pytorch_model.bin",
    'distilbert-base-uncased-distilled-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-pytorch_model.bin"
}

def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer



### UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE ###
def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False

class Embeddings(nn.Module):
    def __init__(self,
                 config):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings,
                                         dim=config.dim,
                                         out=self.position_embeddings.weight)

        self.LayerNorm = LayerNorm(config.dim, eps=1e-12)
        self.dropout = Dropout(config.dropout)
        self.add1 = Add()
        self.add2 = Add()

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.
        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                      # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)                   # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)        # (bs, max_seq_length, dim)

        # embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.add1([word_embeddings, position_embeddings])
        embeddings = self.LayerNorm(embeddings)             # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)               # (bs, max_seq_length, dim)
        return embeddings

    def relprop(self, cam, **kwargs):
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.LayerNorm.relprop(cam, **kwargs)
        # print(cam.shape)
        # [inputs_embeds, position_embeddings, token_type_embeddings]
        (cam1,cam2) = self.add1.relprop(cam, **kwargs)

        return (cam1,cam2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions

        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

        self.matmul1 = MatMul()
        self.matmul2 = MatMul()
        self.softmax = Softmax(dim=-1)
        self.add = Add()
        self.mul = Mul()
        self.head_mask = None
        self.attention_mask = None
        self.clone = Clone()

        self.attn_cam = None
        self.attn = None
        self.attn_gradients = None



    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask, head_mask = None):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)
        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        assert 2 <= mask.dim() <= 3
        causal = (mask.dim() == 3)
        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))           # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))             # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))           # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)                     # (bs, n_heads, q_length, dim_per_head)
        scores = self.matmul1([q, k.transpose(2,3)])          # (bs, n_heads, q_length, k_length)
        mask = (mask==0).view(mask_reshp).expand_as(scores) # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float('inf'))            # (bs, n_heads, q_length, k_length)

        weights = Softmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)        # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = self.matmul2([weights, v])     # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)             # (bs, q_length, dim)
        context = self.out_lin(context)        # (bs, q_length, dim)

        if self.output_attentions:
            return (context, weights)
        else:
            return (context,)

    def relprop(self, cam, **kwargs):
        # Assume output_attentions == False
        cam = self.transpose_for_scores(cam)

        # [attention_probs, value_layer]
        (cam1, cam2) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam2 /= 2
        if self.head_mask is not None:
            # [attention_probs, head_mask]
            (cam1, _)= self.mul.relprop(cam1, **kwargs)


        self.save_attn_cam(cam1)

        cam1 = self.dropout.relprop(cam1, **kwargs)

        cam1 = self.softmax.relprop(cam1, **kwargs)

        if self.attention_mask is not None:
            # [attention_scores, attention_mask]
            (cam1, _) = self.add.relprop(cam1, **kwargs)

        # [query_layer, key_layer.transpose(-1, -2)]
        (cam1_1, cam1_2) = self.matmul1.relprop(cam1, **kwargs)
        cam1_1 /= 2
        cam1_2 /= 2

        # query
        cam1_1 = self.transpose_for_scores_relprop(cam1_1)
        cam1_1 = self.query.relprop(cam1_1, **kwargs)

        # key
        cam1_2 = self.transpose_for_scores_relprop(cam1_2.transpose(-1, -2))
        cam1_2 = self.key.relprop(cam1_2, **kwargs)

        # value
        cam2 = self.transpose_for_scores_relprop(cam2)
        cam2 = self.value.relprop(cam2, **kwargs)

        cam = self.clone.relprop((cam1_1, cam1_2, cam2), **kwargs)

        return cam

class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.dropout = Dropout(p=config.dropout)
        self.lin1 = Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ['relu', 'gelu'], "activation ({}) must be in ['relu', 'gelu']".format(config.activation)
        self.activation = gelu if config.activation == 'gelu' else ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

    def relprop(self, cam, **kwargs):
      cam = self.dropout.relprop(cam,**kwargs)
      cam = self.lin2(cam,**kwargs)
      cam = self.activation.relprop(cam,**kwargs)
      cam = self.lin1.relprop(cam,**kwargs)
      return cam


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = Dropout(p=config.dropout)
        self.activation = config.activation
        self.output_attentions = config.output_attentions

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)
        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask)
        # print(sa_output)
        if self.output_attentions:
            sa_output, sa_weights = sa_output                  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else: # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        # print(sa_output.shape)
        sa_output = self.add2([sa_output,x])
        # print(sa_output.shape)
        sa_output = self.sa_layer_norm(sa_output)          # (bs, seq_length, dim)
        # print(sa_output.shape)
        # Feed Forward Network
        ffn_output = self.ffn(sa_output)                             # (bs, seq_length, dim)
        ffn_output = self.add1([ffn_output,sa_output])
        ffn_output = self.output_layer_norm(ffn_output)  # (bs, seq_length, dim)

        # output = ffn_output
        # print(output.shape)
        # print(sa_weights.shape)
        output = (ffn_output,)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output

    def relprop(self,cam,**kwargs):
      # if self.output_attentions:
      #   camx,cam = cam
      # # cam = (cam
      cam = self.output_layer_norm.relprop(cam,**kwargs)
      cam,cam2 = self.add1.relprop(cam,**kwargs)
      cam = self.ffn.relprop(cam2,**kwargs)

      cam2 = self.sa_layer_norm.relprop(cam2,**kwargs)
      cam2,x = self.add2.relprop(cam2,**kwargs)
      if self.output_attentions:
        cam = (cam, camx)
      else:
        cam = cam[0]
      return self.attention.relprop(cam,**kwargs)






class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        layer = TransformerBlock(config)
        add = Add()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

        self.addh = nn.ModuleList([copy.deepcopy(add) for _ in range(config.n_layers)])
        self.adda = nn.ModuleList([copy.deepcopy(add) for _ in range(config.n_layers)])
        # self.addh1 = Add()
        # self.adda2 = Add()
        # self.add2 = Add()



    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.
        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = ()
        all_attentions = ()

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                # all_hidden_states=self.addh[i]([all_hidden_states,hidden_state])
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(x=hidden_state,
                                         attn_mask=attn_mask,
                                         head_mask=head_mask[i])
            hidden_state = layer_outputs[-1]

            if self.output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                # all_attentions=self.adda[i]([all_attentions,attentions])
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def relprop(self,cam,**kwargs):
      # if self.output_attentions:
      #   # outputs = outputs + (all_hidden_states,)
      #   cam,cam1 = cam
      # if self.output_hidden_states:
      #   cam, cam2 = cam
      # cam3 = cam

      # if self.output_hidden_states:
      #   cam2 , cam3 = cam2 

      for i, layer_module in zip(list(range(5,-1,1)),self.layer[::-1]):
            # if self.output_attentions:
            #   cam1,camx = cam1
            #   camy = camx
            #     # all_attentions = all_attentions + (attentions,)
            # camp= cam3
            # camw=torch.cat([camy,camp])
            # camp,cama,camb=layer_module.relprop(camw,**kwargs)
            cam=layer_module.relprop(cam,**kwargs)
            # if self.output_hidden_states:
            #   cam2,cam3 = cam2

      return cam
      

### INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL ###
class DistilBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = "distilbert"

    def __init__(self, *inputs, **kwargs):
        super(DistilBertPreTrainedModel, self).__init__(*inputs, **kwargs)
    
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, Linear) and module.bias is not None:
            module.bias.data.zero_()


class DistilBertModel(DistilBertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """
    def __init__(self, config):
        super(DistilBertModel, self).__init__(config)

        self.embeddings = Embeddings(config)   # Embeddings
        self.transformer = Transformer(config) # Encoder

        self.init_weights()
        self.add1 = Add()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    def forward(self,
                input_ids, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) # (bs, seq_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids)   # (bs, seq_length, dim)
        # print(embedding_output)
        tfmr_output = self.transformer(x=embedding_output,
                                       attn_mask=attention_mask,
                                       head_mask=head_mask)
        hidden_state = tfmr_output[0]
        # output = self.add1([hidden_state,tfmr_output[1:]])
        output = (hidden_state, ) + tfmr_output[1:]

        return output # last-layer hidden-state, (all hidden_states), (all attentions)

    def relprop(self, cam,**kwargs ):
        # print(cam.shape)
        # (camx, camy) = cam
        # camz = camx
        # camy = torch.cat([camx,camy])
        # cama,camb,camc = self.transformer.relprop(camy,**kwargs)
        cam = self.transformer.relprop(cam,**kwargs)
        cam=self.embeddings.relprop(cam,**kwargs)
        return cam


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    def __init__(self, config):
        super(DistilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.output = Sequential(
            Linear(self.config.hidden_size, 2)  # self.cfg.target_size
        )

        self.init_weights()

    def forward(self, input_ids,  attention_mask=None, head_mask=None, labels=None):
        # distilbert_output = self.distilbert(input_ids=input_ids,
        #                                     attention_mask=attention_mask,
        #                                     head_mask=head_mask)
        

        sequence_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)[0][:, 0, :]
        hidden_states = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)[1]
        attention_weights = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)[-1]
        logits = self.output(sequence_output)
        return logits,attention_weights,hidden_states

        # hidden_state = distilbert_output[0]                    # (bs, seq_len, dim)
        # pooled_output = hidden_state[:, 0]                    # (bs, dim)
        # pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
        # pooled_output = ReLU()(pooled_output)             # (bs, dim)
        # pooled_output = self.dropout(pooled_output)         # (bs, dim)
        # logits = self.classifier(pooled_output)              # (bs, dim)

        # outputs = (logits,) + distilbert_output[1:]
        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs

        # return outputs  # (loss), logits, (hidden_states), (attentions)

    def relprop(self,cam,**kwargs):
      print(cam)
      cam = self.output.relprop(cam,**kwargs)
      print(cam.shape)
      cam = self.distilbert.relprop(cam,**kwargs)
      return cam