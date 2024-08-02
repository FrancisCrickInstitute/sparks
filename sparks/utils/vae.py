from typing import Optional, Union

import torch


def ae_forward(encoder, decoder, inputs, encoder_outputs, tau_p, device, sess_id=0):
    """
    The forward pass of the autoencoder.

    The steps of the forward pass are:
    1. Compute the latent representation of the input.
    2. Stack the latent representation in the encoder_outputs buffer of length tau_p.
    3. Obtain the prediction of the decoder.

    Args:
        encoder (torch.nn.Module): The encoder module of the autoencoder.
        decoder (torch.nn.Module): The decoder module of the autoencoder.
        inputs (torch.tensor): The input data tensor.
        encoder_outputs (torch.tensor): The collection of past encoder output tensors.
        tau_p (int): The size of the past window for the model to consider.
        device (str or torch.device): The device where the tensors will be allocated.
        sess_id (int, optional): The session id. Default is 0.

    Returns:
        encoder_outputs (torch.tensor): The encoder outputs tensor with the newly computed encoder output appended.
        decoder_outputs (torch.tensor): The output tensor of the decoder.
        mu (torch.tensor): The mean of the latent distribution computed by the encoder.
        logvar (torch.tensor): The log-variance of the latent distribution computed by the encoder.
    """

    mu, logvar = encoder(inputs.view(inputs.shape[0], -1).float().to(device), sess_id)
    enc_outputs = encoder.reparametrize(mu, logvar)

    encoder_outputs = torch.cat((encoder_outputs, enc_outputs.unsqueeze(2)), dim=2)
    decoder_outputs = decoder(encoder_outputs[..., -tau_p:], sess_id)

    return encoder_outputs, decoder_outputs, mu, logvar


def skip(encoder: torch.nn.Module,
         inputs: torch.tensor,
         targets: Optional[torch.tensor] = None,
         device: Union[str, torch.device] = 'cpu',
         num_steps: int = 0,
         sess_id: int = 0):
    """
    Runs the model on the first `num_steps` time steps from the input signal to initialise its internal state,
     then returns the remaining signal.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        inputs (torch.tensor): The input data.
        targets (torch.tensor, optional): The target data corresponding to `inputs`.
        device (str or torch.device, optional): The device where the tensor will be allocated. Default is 'cpu'.
        num_steps (int, optional): The number of initial time steps to "burn-in" or skip in the signal.
        sess_id (int, optional): The session id. Default is 0.

    Returns:
        inputs (torch.tensor): The `inputs` tensor after skipping the first `num_steps` time steps.
        targets (torch.tensor): The `targets` tensor after skipping the first `num_steps` time steps
                                if `targets` is not None.
    """

    for t in range(num_steps):
        encoder(inputs[..., t].view(inputs.shape[0], -1).to(device).float(), sess_id)

    inputs = inputs[..., num_steps:]
    if targets is not None:
        targets = targets[..., num_steps:]

    return inputs, targets
