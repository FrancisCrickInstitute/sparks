import os
import pickle

import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from sparks.data.allen.preprocess import preprocess_gratings

output_dir = os.path.join("/nemo/lab/iacarusof/home/shared/Nicolas//datasets/allen_visual/")
manifest_path = os.path.join(output_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session_ids = cache.get_session_table().index.to_numpy()

neuron_types = ['VISp']  # VISp, VISal, VISrl, VISpm, VISam, VISl
min_snr = 1.5
stim_type = 'natural_scenes'
expected_num_stims = 119  # 121 for static_gratings, 119 for natural_scenes

spikes_per_cond, units_ids = preprocess_gratings(cache,
                                                 session_ids,
                                                 stim_type=stim_type,
                                                 neuron_types=neuron_types,
                                                 min_snr=min_snr,
                                                 expected_num_stims=expected_num_stims)

with open(os.path.join(output_dir, "all_spikes_%s_" + stim_type + "_snr_%.1f.pickle") % (neuron_types[0], min_snr), 'wb') as f:
    pickle.dump(spikes_per_cond, f)

units_ids = np.unique(np.concatenate(units_ids))
np.save(os.path.join(output_dir, "units_ids_%s_" + stim_type + "_snr_%.1f.npy") % (neuron_types[0], min_snr), units_ids)
