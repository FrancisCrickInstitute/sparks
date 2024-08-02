import argparse
import os

import numpy as np
import torch
import tqdm
from sklearn.metrics import r2_score

from sparks.data.nlb import make_monkey_reaching_dataset
from sparks.models.controls import LinearEncoder, RNNEncoder
from sparks.models.decoders import get_decoder
from sparks.utils.misc import make_res_folder, save_results
from sparks.utils.test import test
from sparks.utils.train import train_on_batch

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=3, help='Size of the latent space')
    parser.add_argument('--enc_type', type=str, default='linear', choices=['linear', 'rnn'],
                        help='Type of encoder (one of linear or rnn)')
    parser.add_argument('--dec_type', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='Type of decoder (one of linear or mlp)')
    parser.add_argument('--tau_p', type=int, default=10, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')

    # Data parameters
    parser.add_argument('--p_train', type=float, default=0.8,
                        help='Proportion of examples to use for training')
    parser.add_argument('--dt', type=float, default=0.001, help='time bins period')

    args = parser.parse_args()

    make_res_folder('monkey_reaching_control_vae', os.getcwd(), args)

    # Create dataloaders
    (train_dataset, test_dataset,
     train_dl, test_dl) = make_monkey_reaching_dataset(os.path.join(args.home, "datasets/000127/sub-Han/"),
                                                       y_keys='hand_pos',
                                                       batch_size=args.batch_size,
                                                       p_train=args.p_train,
                                                       mode='prediction')

    # Make networks
    if args.enc_type == 'linear':
        encoding_network = LinearEncoder(n_inputs=train_dataset.x_shape,
                                         hidden_dims=[2 * train_dataset.x_shape, 256],
                                         latent_dim=args.latent_dim).to(args.device)
    elif args.enc_type == 'rnn':
        encoding_network = RNNEncoder(n_inputs=train_dataset.x_shape,
                                      hidden_dim=4 * train_dataset.x_shape,
                                      n_layers=1,
                                      latent_dim=args.latent_dim,
                                      device=args.device).to(args.device)
    else:
        raise NotImplementedError

    output_size = train_dataset.y_shape
    decoding_network = get_decoder(output_dim_per_session=output_size * args.tau_f, args=args)

    optimizer = torch.optim.Adam(list(encoding_network.parameters())
                                 + list(decoding_network.parameters()), lr=args.lr)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_test_acc = -np.inf

    for epoch in tqdm.tqdm(range(args.n_epochs)):
        train_iterator = iter(train_dl)
        for inputs, targets in train_iterator:
            train_on_batch(encoder=encoding_network,
                           decoder=decoding_network,
                           inputs=inputs,
                           targets=targets,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           latent_dim=args.latent_dim,
                           tau_p=args.tau_p,
                           tau_f=args.tau_f,
                           beta=args.beta,
                           device=args.device)

        if (epoch + 1) % args.test_period == 0:
            test_loss, encoder_outputs, decoder_outputs = test(encoding_network,
                                                               decoding_network,
                                                               test_dl,
                                                               latent_dim=args.latent_dim,
                                                               tau_p=args.tau_p,
                                                               tau_f=args.tau_f,
                                                               loss_fn=loss_fn,
                                                               device=args.device,
                                                               act=torch.sigmoid)

            test_targets = test_dataset.y_trial_data[..., 100:].transpose(0, 2, 1)
            preds = decoder_outputs[..., 100:].cpu().numpy().transpose(0, 2, 1)
            test_acc = r2_score(test_targets.reshape(-1, test_dataset.y_trial_data.shape[-2]),
                                preds.reshape(-1, test_dataset.y_trial_data.shape[-2]),
                                multioutput='variance_weighted')
            print("Epoch %d, acc: %.3f" % (epoch, test_acc))

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc, encoder_outputs,
                                         decoder_outputs, encoding_network, decoding_network)
