from typing import Any, List

import numpy as np
import torch
import tqdm

from sparks.utils.misc import LongCycler
from sparks.utils.train import train_init, update_and_reset, train_on_batch
from sparks.utils.vae import ae_forward


def train_on_batch_allen_movies_reconstruction(encoder: torch.nn.Module,
                                               decoder: torch.nn.Module,
                                               inputs: torch.tensor,
                                               targets: torch.tensor,
                                               true_frames: torch.tensor,
                                               loss_fn: Any,
                                               optimizer: torch.optim.Optimizer,
                                               latent_dim: int,
                                               tau_p: int,
                                               tau_f: int,
                                               beta: float = 0.,
                                               device: torch.device = 'cpu',
                                               **kwargs):

    """
    Trains the model on a batch of inputs for the reconstruction of Allen movies.
    This function only loads the required frames at every time-step.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        inputs (torch.tensor): The input data for the batch of training.
        targets (torch.tensor): The target frame ids for the batch of training.
        true_frames (torch.tensor): the true frames, targets for reconstruction
        loss_fn (Any): The loss function used to evaluate the model's predictions.
        optimizer (torch.optim.Optimizer): The optimizer algorithm used to update the model's parameters.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): past samples window
        tau_f (int): future samples window
        beta (float, optional): Strength of KLD. Default is 0.
        dt (float, optional): Timestep size. Default is 0.006.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs: Additional keyword arguments.

    Returns:
        None. The model parameters are updated inline.

    """

    online = kwargs.get('online', False)
    sess_id = kwargs.get('sess_id', 0)

    encoder_outputs, loss, T = train_init(encoder, decoder, inputs, latent_dim,
                                          tau_p=tau_p, device=device)

    true_frame = torch.concatenate([true_frames[..., targets[0, t_]].unsqueeze(2).unsqueeze(0)
                                    for t_ in range(tau_f)], dim=-1).repeat_interleave(len(targets), dim=0).to(device)

    for t in range(T - tau_f):
        encoder_outputs, decoder_outputs, mu, logvar = ae_forward(encoder=encoder,
                                                                  decoder=decoder,
                                                                  inputs=inputs[..., t],
                                                                  encoder_outputs=encoder_outputs,
                                                                  tau_p=tau_p,
                                                                  device=device,
                                                                  sess_id=sess_id)

        if online:
            loss = (loss_fn(decoder_outputs, true_frame.flatten(1))
                    - beta * 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
            update_and_reset(encoder, decoder, loss, optimizer)
            encoder.detach_()
            encoder_outputs.detach_()
        else:
            loss += (loss_fn(decoder_outputs, true_frame.flatten(1))
                     - beta * 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        torch.cuda.empty_cache()

        new_frame = true_frames[..., targets[0, t + tau_f]].unsqueeze(2).unsqueeze(0).repeat_interleave(len(targets),
                                                                                                        dim=0)
        true_frame = torch.concatenate([true_frame[..., 1:], new_frame.to(device)], dim=-1)

    if not online:
        update_and_reset(encoder, decoder, loss, optimizer)

    torch.cuda.empty_cache()


def train(encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          train_dls: List,
          loss_fn: Any,
          optimizer: torch.optim.Optimizer,
          latent_dim: int,
          tau_p: int,
          tau_f: int,
          true_frames: torch.tensor = None,
          beta: float = 0.,
          dt: float = 0.006,
          device: torch.device = 'cpu',
          mode: str = 'prediction',
          online: bool = False,
          **kwargs):

    """
    Trains a model for one epoch on the provided dataloaders.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        train_dls (list): A list of data loaders providing train data batches,
        true_frames (torch.tensor): the true frames, targets for reconstruction
        loss_fn (Any): Loss function to evaluate the model's predictions.
        optimizer (torch.optim.Optimizer): The optimizer algorithm to update the model's parameters.
        latent_dim (int): Dimensionality of the latent space.
        tau_p (int): past samples window
        tau_f (int): future samples window
        beta (float, default=0): Strength of KLD.
        dt (float): Timestep size, defaults to 0.006.
        device (torch.device, default='cpu'): The device where the tensors will be allocated.
        mode (str, default='prediction'): The mode of the training process ('prediction', 'unsupervised', etc.).
        online (bool, default=False): A flag indicating if training is performed online.
        **kwargs: Additional keyword arguments.

    Returns:
        None. The model parameters are updated inline.
    """

    start_idx = kwargs.get('start_idx', 0)

    random_order = np.random.choice(np.arange(len(train_dls)), size=len(train_dls), replace=False)
    train_iterator = LongCycler([train_dls[i] for i in random_order])

    for i, (inputs, targets) in enumerate(tqdm.tqdm(train_iterator)):
        if np.isin(mode, ['prediction', 'unsupervised']):
            train_on_batch(encoder=encoder,
                           decoder=decoder,
                           inputs=inputs,
                           targets=targets,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           latent_dim=latent_dim,
                           tau_p=tau_p,
                           tau_f=tau_f,
                           device=device,
                           online=online,
                           beta=beta,
                           sess_id=random_order[i % len(train_dls)] + start_idx,
                           **kwargs)
        # Specific function for reconstruction to avoid loading the film in memory for each example in the minibatch
        elif mode == 'reconstruction':
            train_on_batch_allen_movies_reconstruction(encoder=encoder,
                                                       decoder=decoder,
                                                       inputs=inputs,
                                                       targets=targets,
                                                       true_frames=true_frames,
                                                       loss_fn=loss_fn,
                                                       optimizer=optimizer,
                                                       latent_dim=latent_dim,
                                                       tau_p=tau_p,
                                                       tau_f=tau_f,
                                                       device=device,
                                                       online=online,
                                                       dt=dt,
                                                       beta=beta,
                                                       sess_id=random_order[i % len(train_dls)] + start_idx,
                                                       **kwargs)
        else:
            raise NotImplementedError
