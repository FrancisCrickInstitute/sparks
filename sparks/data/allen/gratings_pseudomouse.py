import os
from typing import Any

import numpy as np
import torch

from sparks.data.allen.utils import sample_correct_unit_ids, load_preprocessed_spikes, make_spike_histogram


def make_gratings_dataset(data_dir: os.path,
                          n_neurons: int = 50,
                          neuron_type: str = 'VISp',
                          min_snr: float = 1.5,
                          dt: float = 0.01,
                          num_examples_train: int = 1,
                          num_examples_test: int = 1,
                          num_workers: int = 0,
                          batch_size: int = 1,
                          correct_units_ids: np.ndarray = None,
                          seed: int = None,
                          target_type: str = 'class'):

    """
    Constructs a dataset for neural response to different grating conditions.

    Parameters
    ----------
    data_dir : os.path
        Path to the directory where the data files are stored.
    n_neurons : int, optional
        Number of neurons to be sampled.
    neuron_type : str, optional
        Type of neurons to sample, defaults to 'VISp'.
    min_snr : float, optional
        Minimum SNR for selecting a neuron.
    dt : float, optional
        Time step.
    num_examples_train : int, optional
        Number of training examples.
    num_examples_test : int, optional
        Number of testing examples.
    num_workers : int, optional
        Number of worker threads for loading the data.
    batch_size : int, optional
        Number of samples per batch.
    correct_units_ids : np.ndarray, optional
        Array containing the IDs of correct neural units, defaults to None.
    seed : int, optional
        Seed for random generators, defaults to None (random seed).
    target_type : str, optional
        Type of target, either 'class' or 'freq'.

    Returns
    -------
    dataset_train: Dataset
        The created training dataset.
    train_dl: DataLoader
        DataLoader object for training data.
    dataset_test: Dataset
        The created test dataset.
    test_dl: DataLoader
        DataLoader object for test data.
    """

    all_spikes, units_ids = load_preprocessed_spikes(data_dir, [neuron_type],
                                                     stim_type='static_gratings', min_snr=min_snr)

    desired_conds = [[4791, 4818, 4862, 4863], [4826, 4831, 4838, 4890],
                     [4788, 4825, 4837, 4858], [4805, 4807, 4892, 4898],
                     [4801, 4835, 4848, 4901]]  # all trial conditions for static gratings grouped by spatial frequency

    # Randomly sample units from preprocessed recording
    correct_units_ids = sample_correct_unit_ids(units_ids, n_neurons, seed, correct_units_ids=correct_units_ids)

    # Get spikes and corresponding targets for each condition
    spikes_train = {unit_id: [] for unit_id in correct_units_ids}
    spikes_test = {unit_id: [] for unit_id in correct_units_ids}
    targets_train = []
    targets_test = []

    if desired_conds is None:
        desired_conds = all_spikes.keys()

    if target_type == 'class':
        targets = np.arange(len(desired_conds))
    elif target_type == 'freq':
        targets = [0.02, 0.04, 0.08, 0.16, 0.32, 0.]
    else:
        raise NotImplementedError

    for target, subconds in zip(targets, desired_conds):
        for subcond in subconds:
            for unit_id in correct_units_ids:
                spikes_train[unit_id].extend([all_spikes[subcond][unit_id][i] for i in range(num_examples_train)])
                spikes_test[unit_id].extend([all_spikes[subcond][unit_id][i] for i in range(num_examples_train,
                                                                                            num_examples_train
                                                                                            + num_examples_test)])
            targets_train.extend([target] * num_examples_train)
            targets_test.extend([target] * num_examples_test)

    targets_train = np.array(targets_train)
    targets_test = np.array(targets_test)

    if len(targets_train) > 0:
        dataset_train = AllenGratingsPseudoMouseDataset(spikes_train, targets_train, correct_units_ids, dt)
        train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    else:
        dataset_train, train_dl = None, None
    if len(targets_test) > 0:
        dataset_test = AllenGratingsPseudoMouseDataset(spikes_test, targets_test, correct_units_ids, dt)
        test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)
    else:
        dataset_test, test_dl = None, None

    return dataset_train, train_dl, dataset_test, test_dl


class AllenGratingsPseudoMouseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 spikes: dict,
                 conds: np.ndarray,
                 good_units_ids: np.ndarray,
                 dt: float = 0.01) -> None:

        """
        Initializes the AllenGratingsPseudoMouseDataset instance.

        Parameters
        ----------
        spikes : dict
            Dictionary containing spike instances.
        conds : np.ndarray
            Array containing the conditions.
        good_units_ids : np.ndarray
            Array containing IDs of good neural units.
        dt : float, optional
            Time step, default is 0.01.
        """

        super(AllenGratingsPseudoMouseDataset).__init__()

        self.dt = dt
        self.good_units_ids = good_units_ids
        self.spikes = spikes
        self.num_neurons = len(good_units_ids)

        self.targets = torch.tensor(conds)
        self.num_targets = len(np.unique(self.targets))

    def __len__(self):
        return len(self.spikes[self.good_units_ids[0]])

    def get_spikes(self, idx):
        """
        Get all spikes for a given presentation of the movie
        """

        time_bin_edges = np.arange(0, 0.25 + self.dt, self.dt)
        return make_spike_histogram(idx, self.good_units_ids, self.spikes, time_bin_edges)

    def __getitem__(self, index: int) -> Any:
        """
        :param: index: int
         Index
        :return: spikes histogram for the given index
        """

        return self.get_spikes(index), self.targets[index]
