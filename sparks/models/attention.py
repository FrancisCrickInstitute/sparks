from typing import Union, List

import numpy as np
import torch
from torch.nn import Parameter


class HebbianAttentionLayer(torch.nn.Module):
    def __init__(self,
                 n_total_neurons: int,
                 embed_dim: int,
                 tau_s: float = 0.5,
                 dt: float = 1.,
                 neurons=None,
                 w_pre: float = 1.,
                 w_post: float = 0.5,
                 data_type: str = 'ephys'):
        """
        HebbianAttentionLayer

        Args:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            tau_s (float, optional): Time constant for the attention. Default is 0.5.
            dt (float, optional): Sampling period. Default is 1.
            neurons: Neurons to be considered.
            If None, a numpy array with shape (n_total_neurons,) is created. Default is None.
            w_pre (float, optional): Initial value for the pre synaptic weights. Default is 1.
            w_post (float, optional): Initial value for the post synaptic weights. Default is 0.5.
            data_type (str, optional): Type of data, can be 'ephys' or 'ca'. Default is 'ephys'.

        Attributes:
            n_total_neurons (int): Total number of neurons.
            embed_dim (int): The embedding dimension.
            neurons: Neurons to be considered.
            attention: The attention feature. Initialized as None.
            data_type (str): Type of data. 'ephys' or 'ca'
            dt (float): Sampling period. Initialized if data_type is 'ephys'.
            tau_s (float): Time constant for the attention. Initialized if data_type is 'ephys'.
            pre_trace: The pre synaptic trace. Initialized if data_type is 'ephys'.
            post_trace: The post synaptic trace. Initialized if data_type is 'ephys'.
            latent_pre_weight: The latent pre synaptic weight. Initialized if data_type is 'ephys'.
            latent_post_weight: The latent post synaptic weight. Initialized if data_type is 'ephys'.
            pre_tau_s: The pre synaptic time constant. Initialized if data_type is 'ephys'.
            post_tau_s: The post synaptic time constant. Initialized if data_type is 'ephys'.
            v_proj (torch.nn.Linear): Linear transformation applied to the attention
                                        feature to make it have the embed_dim dimension.
        """

        super(HebbianAttentionLayer, self).__init__()
        self.n_total_neurons = n_total_neurons

        if neurons is None:
            self.neurons = np.arange(n_total_neurons)
        else:
            self.neurons = neurons

        self.embed_dim = embed_dim
        self.attention = None

        if data_type not in ['ephys', 'calcium']:
            raise NotImplementedError('data_type must be one of ["ephys", "calcium"]')

        self.data_type = data_type
        if data_type == 'ephys':
            self.dt = dt
            self.tau_s = tau_s

            self.pre_trace = None
            self.post_trace = None

            self.latent_pre_weight = None
            self.latent_post_weight = None
            self.pre_tau_s = None
            self.post_tau_s = None
            self.init_latent_weights(w_pre, w_post)

        self.v_proj = torch.nn.Linear(self.n_total_neurons, self.embed_dim)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for the Hebbian Attention layer.

        The forward pass involves calculation of the attention matrix using STDP.
        The calculated attention factor can be likened to the K^T.Q product in conventional dot-product attention.

        Args:
            spikes (torch.Tensor): Tensor representation of the spikes from the neurons.
                                    It should be of shape [n_total_neurons, n_timesteps].

        Returns:
            torch.Tensor: The attention coefficients after applying the attention operation.
            The size of the output tensor is [batch_size, len(n_neurons), embed_dim].
        """

        if self.data_type == 'ephys':
            pre_spikes = spikes.unsqueeze(1)
            post_spikes = spikes[:, self.neurons].unsqueeze(2)

            self.pre_trace_update(pre_spikes)
            self.post_trace_update(post_spikes)

            self.attention = (self.attention
                              + torch.mul(self.pre_trace, post_spikes != 0)
                              - torch.mul(self.post_trace, pre_spikes != 0))

        elif self.data_type == 'calcium':
            pre_spikes = spikes.unsqueeze(1)
            post_spikes = spikes[:, self.neurons].unsqueeze(2)
            self.attention = self.attention + pre_spikes - post_spikes

        return self.v_proj(self.attention) / np.sqrt(self.n_total_neurons + self.embed_dim)

    def pre_trace_update(self, pre_spikes: torch.Tensor) -> None:
        """
        Update the pre-synaptic eligibility traces.

        Args:
            pre_spikes (torch.Tensor): Tensor of pre-synaptic neuron spike activity.

        Returns:
            None
        """
        self.pre_trace = (self.pre_trace * (1 - self.dt / self.pre_tau_s.exp())
                          + (pre_spikes * self.latent_pre_weight.exp()))

    def post_trace_update(self, post_spikes: torch.Tensor) -> None:
        """
        Update the post-synaptic eligibility traces.

        Args:
            post_spikes (torch.Tensor): Tensor of post-synaptic neuron spike activity.

        Returns:
            None
        """
        self.post_trace = (self.post_trace * (1 - self.dt / self.post_tau_s.exp())
                           + (post_spikes * self.latent_post_weight.exp()))

    def init_latent_weights(self, w_pre: float, w_post: float) -> None:
        """
        Initialize the pre and post-synaptic latent weights.
        The traces updates use the exponential of these weights, such that they are positive during the forward
        pass and real-valued for gradients updates.

        The weights are initialized as a zero tensor parameters and then populated with a normal distribution whose mean
        is the logarithm of the corresponding input weight. The standard deviation is the inverse of the square root of
         the sum of total neurons and the length of neurons.

        Args:
            w_pre (float): The value will be used to initialize self.latent_pre_weight.
            w_post (float): The value which will be used to initialize self.latent_post_weight.

        Returns:
            None
        """
        self.latent_pre_weight = Parameter(torch.zeros(1, len(self.neurons), self.n_total_neurons))
        self.latent_post_weight = Parameter(torch.zeros(1, len(self.neurons), self.n_total_neurons))
        torch.nn.init.normal_(self.latent_pre_weight, mean=np.log(w_pre),
                              std=1 / np.sqrt(len(self.neurons) + self.n_total_neurons))
        torch.nn.init.normal_(self.latent_post_weight, mean=np.log(w_post),
                              std=1 / np.sqrt(len(self.neurons) + self.n_total_neurons))

        self.pre_tau_s = Parameter(torch.ones(1, len(self.neurons), self.n_total_neurons) * np.log(self.tau_s))
        self.post_tau_s = Parameter(torch.ones(1, len(self.neurons), self.n_total_neurons) * np.log(self.tau_s))

    def detach_(self):
        """
        Detach the attributes pre_trace, post_trace, and attention from their previous history.

        For 'mps:0' device, this method creates a new tensor by detaching it from the current graph.
        The result will never require gradient. For other devices, this operation will be performed in-place.

        Returns:
            None
        """

        if not hasattr(self.pre_trace, '__iter__'):
            return
        if self.pre_trace.device == torch.device('mps:0'):  # MPS backend doesn't support .detach_()
            self.pre_trace = self.pre_trace.detach()
            self.post_trace = self.post_trace.detach()
            self.attention = self.attention.detach()
        else:
            self.pre_trace.detach_()
            self.post_trace.detach_()
            self.attention.detach_()

    def zero_(self):
        self.pre_trace = 0
        self.post_trace = 0
        self.attention = 0


class MultiHeadedHebbianAttentionLayer(torch.nn.Module):
    def __init__(self,
                 n_total_neurons: int,
                 embed_dim: int,
                 n_heads: int,
                 tau_s: Union[float, List[float]],
                 dt: float = 1,
                 neurons=None,
                 w_pre: float = 1.,
                 w_post: float = 0.5,
                 data_type: str = 'ephys'):
        """
        Initializes the MultiHeadedHebbianAttentionLayer class with given parameters.

        This constructor raises ValueError if embed_dim is not divisible by n_heads
        or if tau_s doesn't have length equal to n_heads.

        Args:
            n_total_neurons (int): The number of total neurons.
            embed_dim (int): The dimension of the embedding space.
            n_heads (int): The number of attention heads.
            tau_s (float or list of floats): A single value or a list of time constants for attention.
            If a single value is provided, it will be replicated for all n_heads.
            dt (float, optional): The timestep size. Default is 1.
            neurons: Neurons of the attention layer. Default is None.
            w_pre (float, optional): The presynaptic weight. Default is 1.
            w_post (float, optional): The postsynaptic weight. Default is 0.5.
            data_type (str, optional): The type of data. Default is 'ephys'.

        Returns:
            None

        Constructs:
            self.heads (torch.nn.ModuleList): A list of attention heads where
             each head is an instance of HebbianAttentionLayer.
        """

        super(MultiHeadedHebbianAttentionLayer, self).__init__()

        if embed_dim % n_heads != 0:
            raise ValueError('embed_dim must be divisible by n_heads')

        if not hasattr(tau_s, '__iter__'):
            tau_s = [tau_s] * n_heads

        if len(tau_s) != n_heads:
            raise ValueError('tau_s must be a list of length n_heads')

        self.heads = torch.nn.ModuleList([HebbianAttentionLayer(n_total_neurons,
                                                                embed_dim // n_heads,
                                                                tau_s=tau_s[i],
                                                                dt=dt,
                                                                neurons=neurons,
                                                                w_pre=w_pre,
                                                                w_post=w_post,
                                                                data_type=data_type)
                                          for i in range(n_heads)])

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention layer.

        Aggregates output of each attention head into a single tensor.
        Passing different values of tau_s allows each attention head to attend to different time-scales.

        Args:
            spikes (torch.Tensor): A 2D tensor containing spikes of the neurons.
                                   The dimension is [batch_size, n_total_neurons].

        Returns:
            attention (torch.Tensor): A 3D tensor containing attention coefficients.
                                      The dimension is [batch_size, n_total_neurons, embed_dim],
                                      where embed_dim is the sum of the output dimensions of all attention heads.
        """

        attention = torch.cat([head(spikes) for head in self.heads], dim=-1)

        return attention

    def detach_(self):
        """
        Detach the attention heads in the current layer from the computational graph.

        No Args.

        No Returns.
        """

        for head in self.heads:
            head.detach_()

    def zero_(self):
        """
        Resets the values of every head in the current layer.

        No Args.

        No Returns.
        """

        for head in self.heads:
            head.zero_()
