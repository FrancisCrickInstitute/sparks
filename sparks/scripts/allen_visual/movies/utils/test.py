from typing import Any, List

import numpy as np
import torch

from sparks.utils.misc import LongCycler
from sparks.utils.test import test_init, test_on_batch
from sparks.utils.vae import ae_forward


@torch.no_grad()
def test_on_batch_allen_movies_prediction(encoder: torch.nn.Module,
                                          decoder: torch.nn.Module,
                                          inputs: torch.tensor,
                                          targets: torch.tensor,
                                          latent_dim: int,
                                          tau_p: int,
                                          tol: int = 30,
                                          device: torch.device = 'cpu',
                                          sess_id: int = 0):

    """
    Tests the model on a batch of inputs and targets using the provided encoder and decoder
    for the prediction of frames from the Allen movie.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        inputs (torch.tensor): The input data for the batch of testing.
        latent_dim (int, optional): The dimensionality of the latent space, defaults to 1.
        tau_p (int, optional): Past window size, defaults to 1.
        tol (int, optional): Tolerance in number of frames, defaults to 30 (corresponding to 1s).
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        sess_id (int, optional): ID of session to choose the correct Hebbian layer from the model, defaults to 0.

    Returns:
        test_acc (float): The testing accuracy on this batch.
                          Accuracy is computed as the fraction of time-steps for which the predicted frame was within
                          1s of the correct one.
        encoder_outputs_batch (torch.Tensor): The outputs of the encoder.
        decoder_outputs_batch (torch.Tensor): The outputs of the decoder.
    """

    encoder_outputs_batch, decoder_outputs_batch, T = test_init(encoder, inputs, latent_dim, tau_p, device)

    for t in range(T):
        encoder_outputs_batch, decoder_outputs, _, _ = ae_forward(encoder, decoder, inputs[..., t],
                                                                  encoder_outputs_batch,
                                                                  tau_p, device,
                                                                  sess_id=sess_id)

        decoder_outputs_batch = torch.cat((decoder_outputs_batch, decoder_outputs.unsqueeze(2)), dim=-1)

    target_windows = [np.arange(t - tol, t + tol)[None, :] for t in targets[0].numpy()]
    test_acc = np.mean(np.array([[np.isin(decoder_outputs_batch[k, :, t].cpu().argmax(dim=-1), target_windows[t])
                                  for t in range(decoder_outputs_batch.shape[-1])]
                                 for k in range(decoder_outputs_batch.shape[0])]))

    return test_acc, encoder_outputs_batch.cpu(), decoder_outputs_batch.cpu()


@torch.no_grad()
def test_on_batch_allen_movies_reconstruction(encoder: torch.nn.Module,
                                              decoder: torch.nn.Module,
                                              inputs: torch.tensor,
                                              targets: torch.tensor,
                                              true_frames: torch.tensor,
                                              latent_dim: int,
                                              tau_p: int,
                                              tau_f: int = 1,
                                              loss_fn: Any = None,
                                              device: torch.device = 'cpu',
                                              sess_id: int = 0):
    """
    Tests the model on a batch of inputs and targets using the provided encoder and decoder
    for the reconstruction of Allen movies.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        inputs (torch.tensor): The input data for the batch of testing.
        targets (torch.tensor): The target frame ids for the batch of testing.
        true_frames (torch.tensor): the true frames, targets for reconstruction
        latent_dim (int, optional): The dimensionality of the latent space, defaults to 1.
        tau_p (int, optional): Past window size, defaults to 1.
        tau_f (int, optional): Future window size, defaults to 1.
        loss_fn (Any, optional): The loss function used to evaluate the model's predictions. Default is None.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        sess_id (int, optional): ID of session to choose the correct Hebbian layer from the model, defaults to 0.

    Returns:
        test_loss (float): The testing loss on this batch.
        encoder_outputs_batch (torch.Tensor): The outputs of the encoder.
        decoder_outputs_batch (torch.Tensor): The outputs of the decoder.
    """

    encoder_outputs_batch, decoder_outputs_batch, T = test_init(encoder, inputs,
                                                                latent_dim, tau_p=tau_p, device=device)
    true_frame = torch.concatenate([true_frames[..., targets[0, t_]].unsqueeze(2).unsqueeze(0)
                                    for t_ in range(tau_f)], dim=-1).repeat_interleave(len(targets), dim=0).to(device)
    test_loss = 0

    for t in range(T):
        encoder_outputs_batch, decoder_outputs, _, _ = ae_forward(encoder, decoder, inputs[..., t],
                                                                  encoder_outputs_batch, tau_p, device, sess_id=sess_id)

        decoder_outputs_batch = torch.cat((decoder_outputs_batch,
                                           torch.sigmoid(decoder_outputs).reshape([len(targets),
                                                                                   true_frames.shape[0],
                                                                                   true_frames.shape[1], -1])), dim=-1)

        if t + tau_f < T:
            test_loss += loss_fn(decoder_outputs, true_frame.flatten(1))
            new_frame = true_frames[..., targets[0, t + tau_f]].unsqueeze(2).unsqueeze(0).repeat_interleave(
                len(targets), dim=0)
            true_frame = torch.concatenate([true_frame[..., 1:], new_frame.to(device)], dim=-1)

    return test_loss, encoder_outputs_batch.cpu(), decoder_outputs_batch.cpu()


@torch.no_grad()
def test(encoder: torch.nn.Module,
         decoder: torch.nn.Module,
         test_dls: List,
         true_frames: torch.tensor,
         mode: str = 'prediction',
         latent_dim: int = 1,
         tau_p: int = 1,
         tau_f: int = 1,
         loss_fn: Any = None,
         device: torch.device = 'cpu',
         **kwargs):

    """
    Tests a model with the specified encoder and decoder on the provided data loaders.
    Specific functions are implemented for prediction and reconstruction to deal with the fact that there are
    several time-steps per frame.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        test_dls (list): A list of data loaders supplying the test data batches.
        true_frames (torch.tensor): the true frames, targets for reconstruction
        mode (str, optional): The mode of the testing process, defaults to 'prediction'.
                              Options include 'prediction', 'reconstruction', and 'unsupervised'.
        latent_dim (int, optional): The dimensionality of the latent space, defaults to 1.
        tau_p (int, optional): Past window size, defaults to 1.
        tau_f (int, optional): Future window size, defaults to 1.
        dt (float, optional): Timestep size, defaults to 0.006.
        loss_fn (Any, optional): The loss function used to evaluate the model's predictions. Default is None.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs: Additional keyword arguments.

    Returns:
        test_acc (float): The testing accuracy.
                          For reconstruction and unsupervised learning, the opposite of the BCE is returned
        encoder_outputs (torch.Tensor): The outputs of the encoder.
        decoder_outputs (torch.Tensor): The outputs of the decoder.
    """

    encoder.eval()
    decoder.eval()

    encoder_outputs = torch.Tensor()
    decoder_outputs = torch.Tensor()

    test_acc = 0

    test_iterator = LongCycler(test_dls)
    sess_ids = kwargs.get('sess_ids', np.arange(len(test_dls)))

    for i, (inputs, targets) in enumerate(test_iterator):
        if mode == 'prediction':
            (acc_batch,
             encoder_outputs_batch,
             decoder_outputs_batch) = test_on_batch_allen_movies_prediction(encoder=encoder,
                                                                            decoder=decoder,
                                                                            inputs=inputs,
                                                                            targets=targets,
                                                                            latent_dim=latent_dim,
                                                                            tau_p=tau_p,
                                                                            device=device,
                                                                            sess_id=sess_ids[i])
            test_acc += acc_batch / np.sum([len(test_dl) for test_dl in test_dls])

        elif mode == 'reconstruction':
            (loss_batch,
             encoder_outputs_batch,
             decoder_outputs_batch) = test_on_batch_allen_movies_reconstruction(encoder=encoder,
                                                                                decoder=decoder,
                                                                                inputs=inputs,
                                                                                targets=targets,
                                                                                true_frames=true_frames,
                                                                                latent_dim=latent_dim,
                                                                                tau_p=tau_p,
                                                                                tau_f=tau_f,
                                                                                loss_fn=loss_fn,
                                                                                device=device,
                                                                                sess_id=sess_ids[i])
            test_acc -= loss_batch.cpu() / np.sum([len(test_dl) for test_dl in test_dls])

        elif mode == 'unsupervised':
            loss_batch, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(encoder=encoder,
                                                                                     decoder=decoder,
                                                                                     inputs=inputs,
                                                                                     targets=inputs,
                                                                                     latent_dim=latent_dim,
                                                                                     tau_p=tau_p,
                                                                                     tau_f=tau_f,
                                                                                     loss_fn=loss_fn,
                                                                                     device=device,
                                                                                     act=torch.sigmoid)
            test_acc -= loss_batch.cpu() / np.sum([len(test_dl) for test_dl in test_dls])
        else:
            raise NotImplementedError

        encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch), dim=0)
        decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch), dim=0)

    return test_acc, encoder_outputs, decoder_outputs
