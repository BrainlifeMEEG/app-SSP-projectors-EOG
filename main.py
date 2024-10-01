
# Copyright (c) 2021 brainlife.io
#
# This file is a MNE python-based brainlife.io App
#


# set up environment
import os
import json
import mne
import helper
import numpy as np
import pandas as pd

#workaround for -- _tkinter.TclError: invalid command name ".!canvas"
# so execution won't hang when figures are shown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Load brainlife config.json
with open('config.json','r') as config_f:
    config = helper.convert_parameters_to_None(json.load(config_f))

# == LOAD DATA ==
fname = config['mne']
raw = mne.io.read_raw_fif(fname, verbose=False)

eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, raw_event=None, tmin=config['tmin'], tmax=config['tmax'], n_grad=config['n_grad'],
            n_mag=config['n_mag'], n_eeg=config['n_eeg'], l_freq=config['l_freq'], h_freq=config['h_freq'], average=config['average'], 
            filter_length=config['filter_length'], 
            n_jobs=-1, ch_name=config['ch_name'], reject=None, flat=None, bads=[],
            avg_ref=config['avg_ref'], no_proj=config['no_proj'], event_id=config['event_id'], eog_l_freq=config['eog_l_freq'], eog_h_freq=config['eog_h_freq'],
            tstart=config['tstart'],
            filter_method=config['filter_method'], iir_params=config['iir_params'], copy=True, return_drop_log=False,
            meg=config['meg'])

mne.write_proj('out_dir/proj.fif', eog_projs, overwrite=True)

# == FIGURES ==
fig_ep = mne.viz.plot_projs_topomap(eog_projs, info=raw.info)
fig_ep.savefig(os.path.join('out_figs','eog_projectors.png'))

eog_evoked = mne.preprocessing.create_eog_epochs(raw).average()
eog_evoked.apply_baseline((None, None))

f = eog_evoked.plot_joint()
[fig.savefig(os.path.join('out_figs',f'eog_{i}.png')) for i, fig in enumerate(f)]

report = mne.Report(title='SSP EOG Projectors')
report.add_projs(info=raw.info, projs=eog_projs, title = 'SSP EOG Projectors')

report.save('out_report/report_ssp.html', overwrite=True)




