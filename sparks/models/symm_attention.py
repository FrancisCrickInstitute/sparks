from typing import Any

import numpy as np
import torch
from torch.nn import ModuleList

from sparks.models.attention import HebbianAttentionLayer, MultiHeadedHebbianAttentionLayer
from sparks.models.encoders import HebbianTransformerBlock, HebbianTransformerEncoder


class SymmetricAttentionLayer(HebbianAttentionLayer):
    """
    Symmetric Attention layer
    """

    def __init__(self, n_total_neurons, embed_dim, tau_s=0.5, dt=1,
                 neurons=None, w_pre=1., w_post=0.5, data_type='ephys'):
        """
        :param n_total_neurons: number of neurons
        :param embed_dim: embedding dimension
        :param tau_s: time constant for the attention
        :param dt: sampling period
        :param neurons: neurons to be considered
        :param w_pre: initial value for the pre synaptic weights
        :param w_post: initial value for the post synaptic weights
        :param data_type: type of data, ephys or calcium
        """

        super(SymmetricAttentionLayer, self).__init__(n_total_neurons, embed_dim, tau_s, dt,
                                                      neurons, w_pre, w_post, data_type)

    def forward(self, spikes):
        """
        Forward pass of the attention layer
        :param spikes: spikes of the neurons [n_total_neurons, n_timesteps]
        :return: attention coefficients [len(n_neurons), embed_dim]
        """

        if self.data_type == 'ephys':
            pre_spikes = spikes.unsqueeze(1)
            post_spikes = spikes[:, self.neurons].unsqueeze(2)

            self.pre_trace_update(pre_spikes)
            self.post_trace_update(post_spikes)

            self.attention = (self.attention
                              + torch.mul(self.pre_trace, post_spikes != 0)
                              + torch.mul(self.post_trace, pre_spikes != 0))

        elif self.data_type == 'calcium':
            pre_spikes = spikes.unsqueeze(1)
            post_spikes = spikes[:, self.neurons].unsqueeze(2)
            self.attention = self.attention + pre_spikes + post_spikes

        return self.v_proj(self.attention) / np.sqrt(self.n_total_neurons + self.embed_dim)


class MultiHeadedSymmetricAttentionLayer(MultiHeadedHebbianAttentionLayer):
    """
    Multi-headed Symmetric Attention layer
    """

    def __init__(self, n_total_neurons, embed_dim, n_heads, tau_s=0.5,
                 dt=1, neurons=None, w_pre=1., w_post=0.5, data_type='ephys'):
        """
        :param tau_s: time constant for the attention
        """
        super(MultiHeadedSymmetricAttentionLayer, self).__init__(n_total_neurons, embed_dim, n_heads, tau_s,
                                                                 dt, neurons, w_pre, w_post, data_type)

        if embed_dim % n_heads != 0:
            raise ValueError('embed_dim must be divisible by n_heads')

        if not hasattr(tau_s, '__iter__'):
            tau_s = [tau_s] * n_heads

        if len(tau_s) != n_heads:
            raise ValueError('tau_s must be a list of length n_heads')

        self.heads = torch.nn.ModuleList([SymmetricAttentionLayer(n_total_neurons,
                                                                  embed_dim // n_heads,
                                                                  tau_s=tau_s[i],
                                                                  dt=dt,
                                                                  neurons=neurons,
                                                                  w_pre=w_pre,
                                                                  w_post=w_post,
                                                                  data_type=data_type)
                                          for i in range(n_heads)])


class SymmetricAttentionBlock(HebbianTransformerBlock):
    """
    Linear encoder with attention layer
    """
    def __init__(self, n_total_neurons,
                 embed_dim,  n_heads=1, tau_s=0.5, dt=1,
                 neurons=None, w_pre=1., w_post=0.5, data_type='ephys'):
        """
        :param n_inputs: number of input neurons
        :param hidden_dims:
        :param latent_dim:
        :param tau_s: time constant for the attention
        :param device: device to use
        """

        super(SymmetricAttentionBlock, self).__init__(n_total_neurons, embed_dim, n_heads, tau_s, dt,
                                                      neurons, w_pre, w_post, data_type)

        self.attention_layer = MultiHeadedSymmetricAttentionLayer(n_total_neurons,
                                                                  embed_dim,
                                                                  n_heads=n_heads,
                                                                  tau_s=tau_s,
                                                                  dt=dt,
                                                                  neurons=neurons,
                                                                  w_pre=w_pre,
                                                                  w_post=w_post,
                                                                  data_type=data_type)


class SymmetricAttentionEncoder(HebbianTransformerEncoder):
    def __init__(self,
                 n_neurons_per_sess: Any,
                 embed_dim: int,
                 latent_dim: int,
                 n_layers: int,
                 output_type: str = 'flatten',
                 n_heads: int = 1,
                 id_per_sess: Any = None,
                 tau_s_per_sess: Any = None,
                 dt_per_sess: Any = None,
                 neurons_per_sess: Any = None,
                 w_pre: float = 1.,
                 w_post: float = 0.5,
                 device: torch.device = torch.device('cpu')):

        """
        :param n_neurons_per_sess: number of input neurons per session
        :param hidden_dims:
        :param latent_dim:
        :param tau_s: time constant for the attention
        :param device: device to use
        """

        super(SymmetricAttentionEncoder, self).__init__(n_neurons_per_sess, embed_dim, latent_dim, n_layers,
                                                        output_type, n_heads, id_per_sess, tau_s_per_sess,
                                                        dt_per_sess, neurons_per_sess, w_pre, w_post, device)

        if not hasattr(n_neurons_per_sess, '__iter__'):
            n_neurons_per_sess = [n_neurons_per_sess]
        if not hasattr(tau_s_per_sess, '__iter__'):
            tau_s_per_sess = [tau_s_per_sess] * len(n_neurons_per_sess)
        if not hasattr(dt_per_sess, '__iter__'):
            dt_per_sess = [dt_per_sess] * len(n_neurons_per_sess)
        if neurons_per_sess is not None:
            if not hasattr(neurons_per_sess, '__iter__'):
                neurons_per_sess = [neurons_per_sess] * len(n_neurons_per_sess)
        else:
            neurons_per_sess = [None] * len(n_neurons_per_sess)

        self.Symmetric_attn_blocks = ModuleList([SymmetricAttentionBlock(n_neurons,
                                                                         embed_dim,
                                                                         n_heads=n_heads,
                                                                         tau_s=tau_s,
                                                                         dt=dt,
                                                                         neurons=neurons,
                                                                         w_pre=w_pre,
                                                                         w_post=w_post)
                                                 for (n_neurons, tau_s, dt, neurons) in zip(n_neurons_per_sess,
                                                                                            tau_s_per_sess,
                                                                                            dt_per_sess,
                                                                                            neurons_per_sess)])
