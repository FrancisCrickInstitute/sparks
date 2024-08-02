import os
from typing import List

import numpy as np
import torch
from nlb_tools.nwb_interface import NWBDataset

from sparks.data.misc import smooth, normalize


def process_dataset(dataset_path: os.path,
                    y_keys: str = 'hand_pos',
                    subkeys: List = None):
    dataset = NWBDataset(dataset_path, "*train", split_heldout=False)

    trial_mask = (~dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')  # only active trials
    unique_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    lag = 40
    align_range = (-100, 500)
    align_field = 'move_onset_time'

    lag_align_range = (align_range[0] + lag, align_range[1] + lag)

    if y_keys == 'direction':
        y_trial_data = [np.ones([len(dataset.make_trial_data(align_field=align_field,
                                                             align_range=lag_align_range,
                                                             ignored_trials=~(trial_mask &
                                                                              (dataset.trial_info['cond_dir'] == angle))
                                                             ).to_numpy()), 1]) * i
                        for i, angle in enumerate(unique_angles)]
    else:
        if subkeys is not None:
            y_trial_data = [dataset.make_trial_data(align_field=align_field,
                                                    align_range=lag_align_range,
                                                    ignored_trials=~(trial_mask &
                                                                     (dataset.trial_info['cond_dir'] == angle)))[
                                y_keys][subkeys].to_numpy() for angle in unique_angles]
        else:
            y_trial_data = [dataset.make_trial_data(align_field=align_field,
                                                    align_range=lag_align_range,
                                                    ignored_trials=~(trial_mask &
                                                                     (dataset.trial_info['cond_dir'] == angle)))[
                                y_keys].to_numpy() for angle in unique_angles]

    align_data = [dataset.make_trial_data(align_field=align_field,
                                          align_range=align_range,
                                          ignored_trials=~(trial_mask & (dataset.trial_info['cond_dir'] == angle)))
                  for angle in unique_angles]

    trial_ids = [align_data[i]['trial_id'].to_numpy() for i in range(len(align_data))]
    unique_ids = [np.unique(trial_ids[i]) for i in range(len(trial_ids))]

    return align_data, y_trial_data, unique_ids, trial_ids


def make_monkey_reaching_dataset(dataset_path: os.path,
                                 y_keys: str = 'hand_pos',
                                 mode: str = 'prediction',
                                 p_train: float = 1.,
                                 batch_size: int = 32,
                                 smooth: bool = False):

    if y_keys == 'force':
        subkeys = ['xmo', 'ymo', 'zmo']
    else:
        subkeys = None

    align_data, y_trial_data, unique_ids, trial_ids = process_dataset(dataset_path, y_keys, subkeys=subkeys)
    normalize_targets = False if y_keys == 'direction' else True

    if mode == 'prediction':
        train_dataset = MonkeyReachingDataset(align_data,
                                              y_trial_data,
                                              unique_ids,
                                              trial_ids,
                                              mode=mode,
                                              p_train=p_train,
                                              smooth=smooth,
                                              train=True,
                                              normalize_targets=normalize_targets)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MonkeyReachingDataset(align_data,
                                             y_trial_data,
                                             unique_ids,
                                             trial_ids,
                                             mode=mode,
                                             p_train=p_train,
                                             smooth=smooth,
                                             train=False,
                                             normalize_targets=normalize_targets)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    else:
        train_dataset = MonkeyReachingDataset(align_data,
                                              y_trial_data,
                                              unique_ids,
                                              trial_ids,
                                              mode=mode,
                                              p_train=1.,
                                              smooth=smooth,
                                              train=True,
                                              normalize_targets=normalize_targets)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MonkeyReachingDataset(align_data,
                                             y_trial_data,
                                             unique_ids,
                                             trial_ids,
                                             mode=mode,
                                             p_train=0.,
                                             smooth=smooth,
                                             train=False,
                                             normalize_targets=normalize_targets)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_dataset, test_dataset, train_dl, test_dl


class MonkeyReachingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            align_data: List,
            y_trial_data: List,
            unique_ids: List,
            trial_ids: List,
            mode: str = 'prediction',
            p_train: float = 1.,
            train: bool = True,
            smooth: bool = False,
            normalize_targets: bool = True,
            p_neurons: float = 1.,
    ) -> None:

        """
        Abstract Dataset for spike encoding

        Parameters
        --------------------------------------

        :param: device : torch.device = 'cpu'
        device to which data is loaded
        """

        super(MonkeyReachingDataset).__init__()

        if train:
            self.indices = [unique_ids[i][:int(len(unique_ids[i]) * p_train)] for i in range(len(unique_ids))]
        else:
            self.indices = [unique_ids[i][int(len(unique_ids[i]) * p_train):] for i in range(len(unique_ids))]

        self.x_trial_data = np.vstack([np.vstack([align_data[i]['spikes'].to_numpy()[trial_ids[i] == idx][None, :]
                                                  for idx in self.indices[i]]) for i in
                                       range(len(align_data))]).transpose([0, 2, 1])
        if normalize_targets:
            self.y_trial_data = normalize(np.vstack([np.vstack([y_trial_data[i][trial_ids[i] == idx][None, :]
                                                                for idx in self.indices[i]]) for i in
                                                     range(len(align_data))]).transpose([0, 2, 1]))
        else:
            self.y_trial_data = np.vstack([np.vstack([y_trial_data[i][trial_ids[i] == idx][None, :]
                                                      for idx in self.indices[i]]) for i in
                                           range(len(align_data))]).transpose([0, 2, 1])

        self.y_shape = self.y_trial_data.shape[-2]
        if p_neurons == 1.:
            self.correct_neurons = np.arange(self.x_trial_data.shape[-2])
        else:
            self.correct_neurons = np.random.choice(self.x_trial_data.shape[-2],
                                                    int(p_neurons * self.x_trial_data.shape[-2]), replace=False)
        self.x_shape = len(self.correct_neurons)
        self.x_trial_data = self.x_trial_data[:, self.correct_neurons]

        self.smooth = smooth
        self.mode = mode

    def __len__(self):
        return len(np.hstack(self.indices))

    def __getitem__(self, index: int):
        """
        :param: index: int
         Index
        :return: tuple: (data, target) where target is index of the target class.
        """

        features = self.x_trial_data[index]
        if self.smooth:
            features = np.vstack([smooth(feature, window=200) for feature in features])

        features = torch.tensor(features).float()
        target = torch.tensor(self.y_trial_data[index]).float()

        if self.mode == 'prediction':
            return features, target
        else:
            return features, features
