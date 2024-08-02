import torch

class LinearEncoder(torch.nn.Module):
    """
    Linear encoder
    """

    def __init__(self, n_inputs, hidden_dims, latent_dim, device='cpu'):
        """
        :param n_inputs: number of input neurons
        :param hidden_dims:
        :param latent_dim:
        :param tau_s: time constant for the attention
        :param device: device to use
        """

        super(LinearEncoder, self).__init__()
        self.n_inputs = n_inputs
        self.device = device

        hidden_dims = [n_inputs] + hidden_dims

        self.layers = torch.nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.fc_mu = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, spikes, session_id=None):
        """
        Forward pass of the encoder
        :param spikes: spikes of the neurons [batch_size, n_neurons, n_timesteps]
        :param session_id for compatibility
        :return: encoded signal the output neurons [batch_size, latent_dim]
        """

        out = spikes.flatten(1)
        for layer in self.layers:
            out = torch.nn.functional.leaky_relu(layer(out))

        mu = self.fc_mu(out)
        logvar = self.fc_var(out)

        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def zero_(self):
        """
        For compatibility
        :return: 
        """
        return


class RNNEncoder(torch.nn.Module):
    """
    RNN encoder
    """

    def __init__(self, n_inputs, hidden_dim, n_layers, latent_dim, device='cpu'):
        """
        :param n_inputs: number of input neurons
        :param hidden_dims:
        :param latent_dim:
        :param tau_s: time constant for the attention
        :param device: device to use
        """

        super(RNNEncoder, self).__init__()
        self.n_inputs = n_inputs
        self.device = device

        self.rnn = torch.nn.RNN(n_inputs, hidden_dim, n_layers, batch_first=True)

        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dim, latent_dim)

        self.hidden_state = None

    def forward(self, spikes, session_id=None):
        """
        Forward pass of the encoder
        :param spikes: spikes of the neurons [batch_size, n_neurons, n_timesteps]
        :return: encoded signal the output neurons [batch_size, latent_dim]
        """

        if self.hidden_state is None:
            self.hidden_state = torch.zeros(self.rnn.num_layers, len(spikes), self.rnn.hidden_size).to(spikes.device)

        out, h = self.rnn(spikes.unsqueeze(1), self.hidden_state)
        self.hidden_state = h

        mu = self.fc_mu(out.flatten(1))
        logvar = self.fc_var(out.flatten(1))

        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def zero_(self):
        """
        For compatibility
        :return:
        """
        self.hidden_state = None
