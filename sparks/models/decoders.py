from typing import Any

import numpy as np
import torch
import torch.nn as nn


def get_decoder(output_dim_per_session: Any,
                args: Any,
                n_neurons_per_session: Any = None,
                id_per_sess: Any = None,
                softmax: bool = False,
                **kwargs):

    if not hasattr(output_dim_per_session, '__iter__'):
        output_dim_per_session = [output_dim_per_session]
    if id_per_sess is None:
        id_per_sess = np.arange(len(output_dim_per_session))

    n_inputs_decoder = args.latent_dim * args.tau_p
    if args.dec_type == 'linear':
        return linear(in_dim=n_inputs_decoder,
                      output_dim_per_session=output_dim_per_session,
                      id_per_sess=id_per_sess,
                      softmax=softmax).to(args.device)
    elif args.dec_type == 'mlp':
        hid_features = kwargs.get('hid_features', int(np.mean([n_inputs_decoder, np.mean(output_dim_per_session)])))
        return mlp(in_dim=n_inputs_decoder,
                   hid_features=hid_features,
                   output_dim_per_session=output_dim_per_session,
                   id_per_sess=id_per_sess,
                   softmax=softmax).to(args.device)
    elif args.dec_type == 'deconv':
        n_neurons_per_session = kwargs.get('n_neurons_per_session', output_dim_per_session)
        return deconv(latent_dim=args.latent_dim,
                      tau_p=args.tau_p,
                      embedding_dim=args.embed_dim,
                      n_neurons_per_session=n_neurons_per_session,
                      output_dim_per_session=output_dim_per_session,
                      id_per_sess=id_per_sess,
                      softmax=softmax).to(args.device)
    else:
        raise ValueError("dec_type must be one of: linear, mlp, deconv")


class mlp(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_features: int,
                 output_dim_per_session: Any,
                 id_per_sess: Any = None,
                 softmax: bool = False) -> None:

        """
        Initialize a Multi-Layer Perceptron (MLP).

        This MLP consists of an input layer, zero or more hidden layers, and an output layer.
        Each layer is a fully connected, or dense, layer, meaning each neuron in one layer is connected to all neurons
        in the previous layer. The last layer has either no activation function or log_softmax.

        If the model is trained to encode more than one session via unsupervised learning, the decoder is given
        one output layer for each to accommodate the varying output dimensions.

        Args:
            in_dim (int): The input dimension of the MLP.
            hid_features (Union[int, List[int]]): The number of hidden neurons in the hidden layers.
            output_dim_per_session (Union[int, List[int]]): The output dimension of the MLP.
            id_per_sess (Optional[np.array]): Defaults to None, the ids of the sessions corresponding to the
                                                various output layers.
            softmax (bool): Defaults to False. If True, apply a log_softmax activation function to the neuron outputs.

        Returns:
            None
        """

        super(mlp, self).__init__()

        if not hasattr(output_dim_per_session, '__iter__'):
            output_dim_per_session = [output_dim_per_session]
        if not hasattr(hid_features, '__iter__'):
            hid_features = [hid_features]
        if id_per_sess is None:
            id_per_sess = np.arange(len(output_dim_per_session))
        self.id_per_sess = id_per_sess

        self.layers = nn.ModuleList()
        self.in_layer = nn.Linear(in_dim, hid_features[0])

        for i in range(len(hid_features) - 1):
            self.layers.append(nn.Linear(hid_features[i], hid_features[i+1]))

        self.out = nn.ModuleList([nn.Linear(hid_features[-1], out_features)
                                        for out_features in output_dim_per_session])
        self.softmax = softmax

    def forward(self, x, sess_id=0) -> torch.Tensor:
        if len(self.id_per_sess) > 1:
            out_layer_idx = np.where(self.id_per_sess == sess_id)[0][0]
        else:
            out_layer_idx = 0

        x = nn.functional.relu(self.in_layer(x.flatten(1)))
        for layer in self.layers:
            x = nn.functional.relu(layer(x))

        if self.softmax:
            return torch.log_softmax(self.out[out_layer_idx](x), dim=-1)
        else:
            return self.out[out_layer_idx](x)


class linear(nn.Module):
    def __init__(self, in_dim: int,
                 output_dim_per_session: Any,
                 id_per_sess: Any = None,
                 softmax: bool = False) -> None:

        """
        Initialize a  fully connected, or dense, layer, with either no activation function or log_softmax.

        If the model is trained to encode more than one session via unsupervised learning, the decoder is given
        one output layer for each to accommodate the varying output dimensions.

        Args:
            in_dim (int): The input dimension of the MLP.
            output_dim_per_session (Union[int, List[int]]): The output dimension of the MLP.
            id_per_sess (Optional[np.array]): Defaults to None, the ids of the sessions corresponding to the
                                                various output layers.
            softmax (bool): Defaults to False. If True, apply a log_softmax activation function to the neuron outputs.

        Returns:
            None
        """

        super(linear, self).__init__()

        if not hasattr(output_dim_per_session, '__iter__'):
            output_dim_per_session = [output_dim_per_session]
        if id_per_sess is None:
            id_per_sess = np.arange(len(output_dim_per_session))
        self.id_per_sess = id_per_sess

        self.fc1 = nn.ModuleList([nn.Linear(in_dim, out_features)
                                        for out_features in output_dim_per_session])
        self.softmax = softmax

    def forward(self, z, sess_id=0) -> torch.Tensor:
        out_layer_idx = np.where(self.id_per_sess == sess_id)[0][0]
        z = self.fc1[out_layer_idx](z.flatten(1))

        if self.softmax:
            return torch.log_softmax(z, dim=-1)
        else:
            return z


class deconv(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 tau_p: int,
                 embedding_dim: int,
                 n_neurons_per_session: int,
                 output_dim_per_session: Any,
                 id_per_sess: Any = None,
                 softmax: bool = False) -> None:

        """
        Initialize a deconvolutional neural network.

        This network mirrors the architecture of the 'conv' encoder.
        The last layer has either no activation function or log_softmax.

        If the model is trained to encode more than one session via unsupervised learning, the decoder is given
        one output layer for each to accommodate the varying output dimensions.

        Args:
            in_dim (int): The input dimension of the MLP.
            hid_features (Union[int, List[int]]): The number of hidden neurons in the hidden layers.
            output_dim_per_session (Union[int, List[int]]): The output dimension of the MLP.
            id_per_sess (Optional[np.array]): Defaults to None, the ids of the sessions corresponding to the
                                                various output layers.
            softmax (bool): Defaults to False. If True, apply a log_softmax activation function to the neuron outputs.

        Returns:
            None
        """

        super(deconv, self).__init__()

        if not hasattr(output_dim_per_session, '__iter__'):
            output_dim_per_session = [output_dim_per_session]
        if id_per_sess is None:
            id_per_sess = np.arange(len(output_dim_per_session))
        self.id_per_sess = id_per_sess

        self.in_layer = nn.ModuleList([nn.Linear(tau_p, int(np.ceil(n_neurons / 64)))
                                        for n_neurons in n_neurons_per_session])

        self.layers = nn.Sequential(nn.ConvTranspose1d(latent_dim, 2 * embedding_dim, kernel_size=1,
                                               stride=1, padding=0, bias=False),
                                     nn.ConvTranspose1d(2 * embedding_dim, 2 * embedding_dim, kernel_size=3,
                                               stride=2, padding=1, output_padding=1, bias=False),
                                    nn.BatchNorm1d(2 * embedding_dim),
                                    nn.ReLU6(inplace=True),
                                    nn.ConvTranspose1d(2 * embedding_dim, 2 * embedding_dim, kernel_size=3,
                                               stride=4, padding=0, output_padding=1, bias=False),
                                    nn.BatchNorm1d(2 * embedding_dim),
                                    nn.ReLU6(inplace=True),
                                    nn.ConvTranspose1d(2 * embedding_dim, embedding_dim // 2, kernel_size=3,
                                               stride=4, padding=0, output_padding=1, bias=False),
                                    nn.BatchNorm1d(embedding_dim // 2),
                                    nn.ReLU6(inplace=True),
                                    nn.ConvTranspose1d(embedding_dim // 2, embedding_dim, kernel_size=3,
                                               stride=2, padding=1, output_padding=1, bias=False),
                                    nn.BatchNorm1d(embedding_dim),
                                    )

        self.softmax = softmax

    def forward(self, x, sess_id=0) -> torch.Tensor:
        if len(self.id_per_sess) > 1:
            layer_idx = np.where(self.id_per_sess == sess_id)[0][0]
        else:
            layer_idx = 0

        x = nn.functional.relu(self.in_layer[layer_idx](x))

        x = self.layers(x).flatten(1)[:, :x.shape[1]]

        if self.softmax:
            return torch.log_softmax(x, dim=-1)
        else:
            return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv2dBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                              stride=stride, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.GELU())

    def forward(self, x):
        return self.layers(x)
