#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:36:52 2021

@author: dejan
"""
import os
import numpy as np
import pandas as pd
from sklearn import decomposition
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import colors
import seaborn as sns
from tkinter import filedialog, Tk, messagebox
from timeit import default_timer as time
from read_WDF import convert_time, read_WDF
from warnings import warn
from utilities import NavigationButtons, slice_lr, baseline_als, AllMaps, long_correction
from CR_search import remove_cosmic_rays

#import deconvolution

sns.set()

initialization = {'SliceValues': [None, None], # Use None to count all
                  'NMF_NumberOfComponents': 6,
                  'PCA_components': 9,
                  # Put in the int number from 0 to _n_y:
                  'NumberOfLinesToSkip_Beggining': 0,
                  # Put in the int number from 0 to _n_y - previous element:
                  'NumberOfLinesToSkip_End': 0,
                  'BaselineCorrection': True,
                  'CosmicRayCorrection': True,
                  # Nearest neighbour method
                  "save_data": False}
#%%
folder_name = "./Data/Chloe/bulles/"
files = os.listdir(folder_name)
file_n = files[1]
filename = folder_name + file_n

# Reading the data from the .wdf file
spectra_raw, sigma, params, map_params, origins =\
                            read_WDF(filename, verbose=True)

n_spectra = params["Count"]
temp_c = origins["Temperature"]["Temperature"].Celcius.iloc[:n_spectra]
titles = [f"Temperature = {T} °C" for T in temp_c]
spectra = np.empty_like(spectra_raw)
for i, T in enumerate(temp_c):
    spectra[i] = spectra_raw[i] * long_correction(sigma, params["LaserWaveLength"], T=T, T0=25)

_s = np.stack((spectra_raw, spectra), axis=-1)
see_long_correction = NavigationButtons(sigma, _s, autoscale_y=True,
                                        title=titles,
                                        label=["raw", "corrected"])
see_long_correction.figr.suptitle("Long Correction")
#%%
# =============================================================================
#                               SLICING....
# =============================================================================
# Isolating the part of the spectra that interests us
try:
    pos_left = initialization["SliceValues"][0]
except (ValueError, TypeError, KeyError):
    pos_left = None
try:
    pos_right = initialization["SliceValues"][1]
except (ValueError, TypeError, KeyError):
    pos_right = None

spectra_kept, sigma_kept = slice_lr(spectra, sigma,
                                    pos_left=pos_left,
                                    pos_right=pos_right)


# =============================================================================
#                               BASELINE....
# =============================================================================
if initialization['BaselineCorrection']:
    _start = time()
    print("starting the baseline correction..."
          "\n(be patient, this may take some time...)")
    b_line = baseline_als(spectra_kept, p=1e-3, lam=1000*len(sigma_kept))
    _end = time()
    print(f"baseline correction done in {_end - _start:.2f}s")
    # Remove the eventual offsets:
    b_corr_spectra = spectra_kept - b_line
    b_corr_spectra -= np.min(b_corr_spectra, axis=1)[:, np.newaxis]

    # Visualise the baseline correction:
    _baseline_stack = np.stack((spectra_kept, b_line, b_corr_spectra),
                               axis=-1)
    labels = ['original spectra', 'baseline', 'baseline corrected spectra']
    check_baseline = NavigationButtons(sigma_kept, _baseline_stack, title=titles,
                                       autoscale_y=True, label=labels)
    plt.suptitle("baseline")
else:
    b_corr_spectra = spectra_kept -\
                    np.min(spectra_kept, axis=-1, keepdims=True)

# %%
# =============================================================================
#                                 CR correction...
# =============================================================================

mock_sp3 = remove_cosmic_rays(b_corr_spectra,
                              sigma_kept, sensitivity=1)[0].squeeze()

#%%
# =============================================================================
# ---------------------------------- PCA --------------------------------------
# =============================================================================
print(f"smoothing with PCA ({initialization['PCA_components']} components)")
# =============================================================================
mock_sp3 /= (1+np.max(mock_sp3, axis=-1, keepdims=True))
pca = decomposition.PCA(n_components=initialization['PCA_components'])
spectra_reduced = pca.fit_transform(mock_sp3)
# spectra_reduced = np.dot(mock_sp3 - np.mean(mock_sp3, axis=0), pca.components_.T)

spectra_denoised = pca.inverse_transform(spectra_reduced)
# spectra_denoised = np.dot(spectra_reduced, pca.components_)+np.mean(mock_sp3, axis=0)


# =============================================================================
# #%%
# sq_err = (mock_sp3-spectra_denoised)
# vidji_pca_err = AllMaps(sq_err.reshape(_n_yy, _n_x, -1), sigma=sigma_kept,
#                 title="denoising error")
# =============================================================================

########### showing the smoothed spectra #####################
_s = np.stack((mock_sp3,
               spectra_denoised), axis=-1)
see_all_denoised = NavigationButtons(sigma_kept, _s, autoscale_y=True,
                                    label=["scaled orig spectra",
                                           "pca denoised"],
                                    title=titles,
                                    figsize=(12, 12))
see_all_denoised.figr.suptitle("PCA denoising result")


# _pca_components_stack = np.stack(tuple(pca.components_), axis=-1)
see_pca = NavigationButtons(sigma_kept, pca.components_, autoscale_y=True)
see_pca.figr.suptitle("PCA components")

# %%
# =============================================================================
#                                   NMF step
# =============================================================================

#spectra_cleaned = clean(sigma_kept, b_corr_spectra, mode='area')
spectra_cleaned = spectra_denoised - np.min(spectra_denoised, axis=-1, keepdims=True)
_n_components = initialization['NMF_NumberOfComponents']
nmf_model = decomposition.NMF(n_components=_n_components, init='nndsvda',
                              max_iter=1777, l1_ratio=1)
_start = time()
#print('starting nmf... (be patient, this may take some time...)')
mix = nmf_model.fit_transform(spectra_cleaned)
components = nmf_model.components_
reconstructed_spectra = nmf_model.inverse_transform(mix)
_end = time()
print(f'nmf done in {_end - _start:.2f}s')
#%%
see_nmf = NavigationButtons(sigma_kept, components, autoscale_y=True)
see_nmf.figr.suptitle("NMF analysis")
#%%
from scipy.interpolate import interp1d

ttt = temp_c.to_numpy()
# If we want to have equal number of points heating vs cooling:
corr_temp = ttt[55:][::-1]
mix /= np.sum(mix, axis=1, keepdims=True)
mix_cool = mix.T[:, 55:][:,::-1]
mix_heat = mix.T[:, :56]
corr_mix_heat = np.empty_like(mix_cool)
for ii, mixx in enumerate(mix_heat):
    f = interp1d(ttt[:56], mixx, fill_value=0, bounds_error=False)
    corr_mix_heat[ii] = f(corr_temp)
heat_cool_mix = np.stack((mix_cool, corr_mix_heat), axis=-1)

see_evolution = NavigationButtons(corr_temp, heat_cool_mix, autoscale_y=True,
                                  label=["cooling", "heating"], title="Component")
see_evolution.axr.set(ylabel="Component's contribution",
                      xlabel="Temperature [°C]")
#my_xticks = see_evolution.axr.get_xticks()
#my_ticklabels = [str(temp_c.get(xt, ''))+' °C' for xt in my_xticks]
#see_evolution.axr.set_xticklabels(my_ticklabels)