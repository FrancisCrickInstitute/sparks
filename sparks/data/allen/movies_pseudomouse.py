import os
from typing import List

import numpy as np
import torch
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from sparks.data.allen.utils import AllenMoviesNpxDataset, AllenMoviesCaDataset
from sparks.data.allen.utils import (make_spikes_dict, sample_correct_unit_ids,
                                     load_preprocessed_spikes, get_train_test_indices)


def make_npx_dataset(data_dir: os.path,
                     n_neurons: int = 50,
                     neuron_types: List = ['VISp'],
                     mode='prediction',
                     min_snr: float = 1.5,
                     dt: float = 0.01,
                     ds: int = 1,
                     block: str = 'first',
                     num_workers: int = 0,
                     batch_size: int = 1,
                     correct_units_ids: np.ndarray = None,
                     seed: int = None):
    """
    Constructs a dataset for neural experiments.

    Parameters
    ----------
    data_dir : os.path
        Path to the directory where the data files are stored.
    n_neurons : int, optional
        Number of neurons to be sampled.
    neuron_types : list, optional
        List of neuron types to sample, defaults to ['VISp'].
    mode : str, optional
        Mode of operation, either 'prediction' or 'reconstruction'.
    min_snr : float, optional
        Minimum SNR for selecting a neuron.
    dt : float, optional
        Time step.
    ds : int, optional
        Downsampling factor.
    block : str, optional
        Specifies which part of the movie to take ('first' or 'second'), defaults to 'first'.
    num_workers : int, optional
        Number of worker threads for loading the data.
    batch_size : int, optional
        Number of samples per batch.
    correct_units_ids : np.ndarray, optional
        Predefined set of correct unit ids, defaults to None.
    seed : int, optional
        Seed for random number generators, defaults to None (random seed).

    Returns
    -------
    train_dataset : Dataset
        The created training dataset.
    test_dataset : Dataset
        The created test dataset.
    train_dl : DataLoader
        DataLoader for training data.
    test_dl : DataLoader
        DataLoader for testing data.
    """

    all_spikes, units_ids = load_preprocessed_spikes(data_dir, neuron_types,
                                                     stim_type='natural_movie_one', min_snr=min_snr)

    # Randomly sample units from preprocessed recording
    correct_units_ids = sample_correct_unit_ids(units_ids, n_neurons, seed, correct_units_ids=correct_units_ids)

    if mode == 'reconstruction':
        manifest_path = os.path.join(data_dir, "manifest.json")
        cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    else:
        cache = None

    train_indices, test_indices = get_train_test_indices(block, mode)

    train_dataset = AllenMoviesNpxDataset(make_spikes_dict(all_spikes, train_indices, correct_units_ids),
                                          correct_units_ids, dt, cache=cache, ds=ds, mode=mode)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = AllenMoviesNpxDataset(make_spikes_dict(all_spikes, test_indices, correct_units_ids),
                                         correct_units_ids, dt, cache=cache, ds=ds, mode=mode)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, test_dataset, train_dl, test_dl


def make_ca_dataset(n_neurons: int = 50,
                    num_workers: int = 0,
                    batch_size: int = 1,
                    mode='prediction',
                    dt: float = 0.01,
                    ds: int = 1,
                    seed: int = None):
    """
    Constructs a dataset for calcium imaging experiments.

    Parameters
    ----------
    n_neurons : int, optional
        Number of neurons to be sampled, defaults to 50.
    num_workers : int, optional
        Number of worker threads for loading the data, defaults to 0.
    batch_size : int, optional
        Number of samples per batch, defaults to 1.
    mode : str, optional
        Mode of operation, either 'prediction' or 'reconstruction', defaults to 'prediction'.
    dt : float, optional
        Time interval between samples (in seconds), defaults to 0.01.
    ds : int, optional
        Downsampling factor, defaults to 1.
    seed : int, optional
        Seed for the random number generator, defaults to None.

    Returns
    -------
    train_dataset : AllenMoviesCaDataset
        The created training dataset.
    test_dataset : AllenMoviesCaDataset
        The created test dataset.
    train_dl : DataLoader
        DataLoader for training data.
    test_dl : DataLoader
        DataLoader for testing data.
    """

    train_dataset = AllenMoviesCaDataset(n_neurons, seed, train=True, dt=dt, ds=ds, mode=mode)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

    test_dataset = AllenMoviesCaDataset(n_neurons, seed, train=False, dt=dt, ds=ds, mode=mode)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

    return train_dataset, test_dataset, train_dl, test_dl


def make_pseudomouse_allen_movies_dataset(data_dir: os.path = '/',
                                          n_neurons: int = 50,
                                          neuron_types: List = ['VISp'],
                                          mode='prediction',
                                          block: str = 'within',
                                          min_snr: float = 1.5,
                                          dt: float = 0.01,
                                          ds: int = 1,
                                          num_workers: int = 0,
                                          batch_size: int = 1,
                                          correct_units_ids: np.ndarray = None,
                                          data_type: str = 'npx',
                                          seed: int = None):
    """
    Creates a pseudomouse dataset based on Allen's movies.

    Parameters
    ----------
    data_dir : os.path, optional
        Path to the directory where the data files are stored, defaults to '/'.
    n_neurons : int, optional
        Number of neurons to be sampled, defaults to 50.
    neuron_types : list, optional
        List of neuron types to be included, defaults to ['VISp'].
    mode : str, optional
        Mode of operation, either 'prediction' or 'reconstruction', defaults to 'prediction'.
    block : str, optional
        Which part of the movie to take ('first', 'second', or 'within'), defaults to 'within'.
    min_snr : float, optional
        Minimum SNR for selecting a neuron (only used when data_type is 'npx'), defaults to 1.5.
    dt : float, optional
        Time interval between samples (in seconds), defaults to 0.01.
    ds : int, optional
        Downsampling factor, defaults to 1.
    num_workers : int, optional
        Number of worker threads for loading the data, defaults to 0.
    batch_size : int, optional
        Number of samples per batch, defaults to 1.
    correct_units_ids : np.ndarray, optional
        Predefined set of correct unit ids (only used when data_type is 'npx'), defaults to None.
    data_type : str, optional
        Type of data to create the dataset from, either 'npx' or 'ca', defaults to 'npx'.
    seed : int, optional
        Seed for the random number generator, defaults to None.

    Returns
    -------
    The created dataset and dataloaders,
    return value will be as returned by `make_npx_dataset` or `make_ca_dataset` function.
    """

    if data_type == 'npx':
        return make_npx_dataset(data_dir, n_neurons, neuron_types, mode, min_snr, dt, ds,
                                block, num_workers, batch_size, correct_units_ids, seed)
    elif data_type == 'ca':
        return make_ca_dataset(n_neurons, num_workers, batch_size, mode=mode,
                               dt=dt, ds=ds, seed=seed)
