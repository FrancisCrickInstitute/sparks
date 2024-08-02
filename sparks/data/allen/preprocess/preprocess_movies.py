import os
import pickle

import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from sparks.data.allen.preprocess.utils import get_movies_data_for_session

output_dir = os.path.join("/nemo/lab/iacarusof/home/shared/Nicolas//datasets/allen_visual/")
manifest_path = os.path.join(output_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session_ids = cache.get_session_table().index.to_numpy()

all_spikes = {}
units_ids = []

neuron_types = ['VISp', 'VISal', 'VISrl', 'VISpm', 'VISam', 'VISl'] # ['VISp']
stim_type = 'natural_movie_one'
min_snr = 1.5

for session_id in session_ids:
    spikes_session = get_movies_data_for_session(cache, session_id, neuron_types, stim_type, min_snr)
    if spikes_session is not None:
        for idx in spikes_session.keys():
            all_spikes[str(session_id) + str(idx)] = spikes_session[idx]
            units_ids.append(str(session_id) + str(idx))

with open(os.path.join(output_dir, "all_spikes_all_natural_movie_one_snr_1.5.pickle"), 'wb') as f:
    pickle.dump(all_spikes, f)

np.save(os.path.join(output_dir, "units_ids_all_natural_movie_one_snr_1.5.npy"), np.array(units_ids))
