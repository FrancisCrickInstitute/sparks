from itertools import chain
from typing import Any, List

import numpy as np
import torch


def get_spike_times(all_spike_times, start_stop_times):
    return [np.array(list(chain.from_iterable([spike_times_cluster[np.where((spike_times_cluster
                                                                             >= start_stop_times[i, 0])
                                                                            & (spike_times_cluster[i]
                                                                               <= start_stop_times[i, -1]))]
                                               for i in range(len(start_stop_times))])))
            for spike_times_cluster in all_spike_times]


def make_histogram(spikes, start_time, stop_time, dt):
    time_bin_edges = np.arange(0, stop_time - start_time + dt, dt)
    frame_histogram = []

    for spikes_unit in spikes:
        valid_spikes = spikes_unit[(spikes_unit > start_time) & (spikes_unit < stop_time)] - start_time
        if len(valid_spikes) > 0:
            frame_histogram.append(np.histogram(valid_spikes, bins=time_bin_edges)[0] > 0)
        else:
            frame_histogram.append(np.zeros_like(time_bin_edges[:-1]))

    return np.vstack(frame_histogram)


def make_mec_dataset(nwbfile,
                     start_stop_times_train: np.ndarray,
                     start_stop_times_test: np.ndarray,
                     topk: int = 100,
                     dt: float = 0.01,
                     num_workers: int = 0,
                     batch_size: int = 1):
    """
    Creates datasets and data loaders for the MEC data.

    Parameters
    --------------------------------------

    :return:
        train_dataset, test_dataset, train_dl, test_dl
    """

    clusters_quality = nwbfile.units['cluster_quality'][()]
    good_clusters = np.where(clusters_quality == 'good')[0]

    spike_times = nwbfile.units['spike_times']
    spike_times_good_clusters = [spike_times[i] for i in good_clusters]

    end_time = np.max([np.max(spike_times_cluster) for spike_times_cluster in spike_times_good_clusters])
    all_spikes_histogram = make_histogram(spike_times_good_clusters, 0, end_time, dt)

    std_per_neuron = np.nan_to_num(np.std(all_spikes_histogram, axis=-1))

    top_ind = np.argpartition(std_per_neuron, -topk)[-topk:]
    spike_times_topk_clusters = [spike_times_good_clusters[k] for k in top_ind]

    spike_times_train = get_spike_times(spike_times_topk_clusters, start_stop_times_train)
    train_dataset = MECDataset(spike_times_train, start_stop_times_train, dt)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

    spike_times_test = get_spike_times(spike_times_topk_clusters, start_stop_times_test)
    test_dataset = MECDataset(spike_times_test, start_stop_times_test, dt)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

    return train_dataset, train_dl, test_dataset, test_dl


class MECDataset(torch.utils.data.Dataset):
    def __init__(self,
                 spikes: List,
                 start_stop_times: np.ndarray,
                 dt: float = 0.01) -> None:
        """
        Dataset class for MEC data

        Parameters
        --------------------------------------
        :param spikes: dict
        :param start_stop_times: np.ndarray
        :param dt: float
        """

        super(MECDataset).__init__()

        self.dt = dt
        self.start_stop_times = start_stop_times
        self.spikes = spikes

    def __len__(self):
        """ """
        return len(self.start_stop_times)

    def get_spikes(self, idx):
        """
        Make spikes histogram for a given time interval
        """

        start_time, stop_time = self.start_stop_times[idx]
        return make_histogram(self.spikes, start_time, stop_time, self.dt)

    def __getitem__(self, index: int) -> Any:
        """
        :param: index: int
         Index
        :return: spikes histogram for the given index
        """

        return self.get_spikes(index)
