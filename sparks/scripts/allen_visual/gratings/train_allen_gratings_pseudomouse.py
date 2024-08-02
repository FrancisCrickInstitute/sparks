import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.allen.gratings_pseudomouse import make_gratings_dataset
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.misc import make_res_folder, save_results
from sparks.utils.test import test_on_batch
from sparks.utils.train import train_on_batch

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.01, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=128, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--dec_type', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='Type of decoder (one of linear or mlp)')
    parser.add_argument('--tau_p', type=int, default=10, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=0.5, help='STDP decay')

    # data
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--target_type', type=str, default='freq', choices=['freq', 'class'],
                        help='Type of target to predict: either spatial frequencies or class index')
    parser.add_argument('--num_examples_train', type=int, default=30, help='Number of training example')
    parser.add_argument('--num_examples_test', type=int, default=10, help='Number of test example')
    parser.add_argument('--dt', type=float, default=0.001, help='time bins period')
    parser.add_argument('--neuron_type', type=str, default='VISp',
                        choices=['VISp', 'VISal', 'VISrl', 'VISpm', 'VISam', 'VISl'])
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

    args = parser.parse_args()

    make_res_folder('allen_gratings_pseudomouse_' + args.target_type, os.getcwd(), args)

    (train_dataset, train_dl,
     test_dataset, test_dl) = make_gratings_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                    n_neurons=args.n_neurons,
                                                    dt=args.dt,
                                                    neuron_type=args.neuron_type,
                                                    num_examples_train=args.num_examples_train,
                                                    num_examples_test=args.num_examples_test,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    target_type=args.target_type,
                                                    seed=args.seed)
    np.save(args.results_path + '/good_units_ids.npy', train_dataset.good_units_ids)

    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=len(train_dataset.good_units_ids),
                                                 embed_dim=args.embed_dim,
                                                 latent_dim=args.latent_dim,
                                                 tau_s_per_sess=args.tau_s,
                                                 dt_per_sess=args.dt,
                                                 n_layers=args.n_layers,
                                                 n_heads=args.n_heads).to(args.device)

    if args.target_type == 'freq':
        output_size = 1
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.target_type == 'class':
        output_size = 5
        loss_fn = torch.nn.NLLLoss()
    else:
        raise NotImplementedError

    decoding_network = get_decoder(output_dim_per_session=output_size * args.tau_f, args=args)

    if args.online:
        args.lr = args.lr / (0.25 * args.dt)
    optimizer = torch.optim.Adam(list(encoding_network.parameters())
                                 + list(decoding_network.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.9)

    loss_best = np.inf

    for epoch in tqdm.tqdm(range(args.n_epochs)):
        train_iterator = iter(train_dl)
        for inputs, targets in train_iterator:
            targets = targets.unsqueeze(1).repeat_interleave(inputs.shape[-1], dim=-1)
            train_on_batch(encoder=encoding_network,
                           decoder=decoding_network,
                           inputs=inputs,
                           targets=targets,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           latent_dim=args.latent_dim,
                           tau_p=args.tau_p,
                           tau_f=args.tau_f,
                           device=args.device,
                           online=args.online,
                           beta=args.beta)

        if (epoch + 1) % args.test_period == 0:
            test_iterator = iter(test_dl)
            encoder_outputs = torch.Tensor()
            decoder_outputs = torch.Tensor()
            test_loss = 0

            for inputs, targets in test_iterator:
                targets = targets.unsqueeze(1).repeat_interleave(inputs.shape[-1], dim=-1)
                test_loss, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(encoder=encoding_network,
                                                                                        decoder=decoding_network,
                                                                                        inputs=inputs,
                                                                                        targets=targets,
                                                                                        latent_dim=args.latent_dim,
                                                                                        tau_p=args.tau_p,
                                                                                        tau_f=args.tau_f,
                                                                                        test_loss=test_loss,
                                                                                        loss_fn=loss_fn,
                                                                                        device=args.device,
                                                                                        act=torch.sigmoid)

                encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch), dim=0)
                decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch), dim=0)

            print('Avg test loss: ', test_loss)
            loss_best = - save_results(args.results_path, -test_loss.cpu(), loss_best, encoder_outputs,
                                       decoder_outputs, encoding_network, decoding_network)
