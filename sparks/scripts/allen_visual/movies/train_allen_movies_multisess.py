import argparse
import os

import numpy as np
import torch

from sparks.data.allen.movies_singlesess import make_allen_movies_dataset
from sparks.scripts.allen_visual.movies.utils.test import test
from sparks.scripts.allen_visual.movies.utils.train import train
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.misc import make_res_folder, save_results

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--beta', type=float, default=0.001, help='KLD regularisation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=128, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of conventional attention layers')
    parser.add_argument('--dec_type', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='Type of decoder (one of linear or mlp)')
    parser.add_argument('--tau_p', type=int, default=6, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=0.5, help='STDP decay')

    # Data parameters
    parser.add_argument('--n_skip_sessions', type=int, default=0, help='First session to consider')
    parser.add_argument('--n_sessions', type=int, default=1, help='How many sessions to use')
    parser.add_argument('--block', type=str, default='first',
                        choices=['first', 'second', 'across', 'both'], help='From which blocks to use')
    parser.add_argument('--mode', type=str, default='prediction',
                        choices=['prediction', 'reconstruction', 'unsupervised'],
                        help='Which type of task to perform')
    parser.add_argument('--dt', type=float, default=0.006, help='Time sampling period')
    parser.add_argument('--ds', type=int, default=2, help='Frame downsampling factor')

    args = parser.parse_args()

    # Create folder to save results
    make_res_folder('allen_movies_multisess', os.getcwd(), args)

    # Create datasets and dataloaders
    (train_datasets, test_datasets,
     train_dls, test_dls) = make_allen_movies_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                      session_idxs=np.arange(args.n_skip_sessions,
                                                                             args.n_sessions + args.n_skip_sessions),
                                                      dt=args.dt,
                                                      block=args.block,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.num_workers,
                                                      mode=args.mode,
                                                      ds=args.ds)

    # Create encoding/decoding networks
    input_sizes = [len(train_dataset.good_units_ids) for train_dataset in train_datasets]
    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=input_sizes,
                                                 embed_dim=args.embed_dim,
                                                 latent_dim=args.latent_dim,
                                                 tau_s_per_sess=args.tau_s,
                                                 dt_per_sess=args.dt,
                                                 n_layers=args.n_layers,
                                                 n_heads=args.n_heads,
                                                 w_pre=0.5).to(args.device)

    if args.mode == 'prediction':
        output_sizes = 900 * args.tau_f
    elif args.mode == 'reconstruction':
        output_sizes = [np.prod(train_dataset.targets.shape[:-1]) * args.tau_f for train_dataset in train_datasets]
    elif args.mode == 'unsupervised':
        output_sizes = [input_size * args.tau_f for input_size in input_sizes]
    else:
        raise NotImplementedError

    decoding_network = get_decoder(output_dim_per_session=output_sizes, args=args,
                                   softmax=True if args.mode == 'prediction' else False)

    if args.online:
        args.lr = args.lr / 900
    optimizer = torch.optim.Adam(list(encoding_network.parameters())
                                 + list(decoding_network.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    if args.mode == 'prediction':
        loss_fn = torch.nn.NLLLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best_test_acc = -np.inf

    for epoch in range(args.n_epochs):
        train(encoder=encoding_network, decoder=decoding_network, train_dls=train_dls, loss_fn=loss_fn,
              optimizer=optimizer, latent_dim=args.latent_dim, tau_p=args.tau_p, mode=args.mode,
              tau_f=args.tau_f, device=args.device, dt=args.dt, online=args.online, beta=args.beta)
        scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            test_acc, encoder_outputs, decoder_outputs = test(encoder=encoding_network,
                                                              decoder=decoding_network,
                                                              test_dls=test_dls,
                                                              mode=args.mode,
                                                              latent_dim=args.latent_dim,
                                                              tau_p=args.tau_p,
                                                              tau_f=args.tau_f,
                                                              dt=args.dt,
                                                              loss_fn=loss_fn,
                                                              device=args.device)

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc, encoder_outputs,
                                         decoder_outputs, encoding_network, decoding_network)
            if args.mode == 'prediction':
                print("Epoch %d, test acc: %.3f" % (epoch, test_acc))
            else:
                print("Epoch %d, test loss: %.3f" % (epoch, -test_acc))
