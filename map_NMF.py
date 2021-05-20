# -*- coding: utf-8 -*-
# %%
import os
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.neighbors import LocalOutlierFactor
from scipy.signal import savgol_filter, correlate2d
from scipy.ndimage import median_filter
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from matplotlib.widgets import Button
import seaborn as sns
from tkinter import filedialog, Tk, messagebox
from timeit import default_timer as time
from read_WDF import convert_time, read_WDF
from utilities import NavigationButtons, clean, rolling_median,\
slice_lr, baseline_als
#import deconvolution

sns.set()
'''This script uses NMF deconvolution
to produce some informative graphical output on map scans.
ATTENTION: you are supposed to provide your spectra in .wdf files)
All of the imported scripts should be in your working directory

You should first choose the data file with the map scan in the .wdf format

Set the initialization dictionary values
That'_s it!
First plot: your spectra (with navigation buttons)
Second plot: the components found
Third plot: the heatmap of the mixing coefficients
            (shows the abundance of each component troughout the map)
            when you double-click on a pixel on this map,
            it will pop-up another plot
            showing the spectra recorded at this point,
            together with the contributions of each component
'''
#%%
# -----------------------Choose a file-----------------------------------------
filename = 'Data/Sirine/carto - 2h_Copy.wdf'

initialization = {'SliceValues': None,  # [100, 1300],  # Use None to count all
                  'NMF_NumberOfComponents': 6,
                  'PCA_components': 0.998,
                  # Put in the int number from 0 to _n_y:
                  'NumberOfLinesToSkip_Beggining': 0,
                  # Put in the int number from 0 to _n_y - previous element:
                  'NumberOfLinesToSkip_End': 0,
                  'BaselineCorrection': True,
                  'CosmicRayCorrection': True,
                  'AbsoluteScale': False}  # what type of colorbar to use

# Reading the data from the .wdf file
spectra, sigma, params, map_params, origins =\
                            read_WDF(filename, verbose=True)


'''
- **"spectra"** is a 2D numpy array containing the intensities
    recorded at each point in a map scan.
    It is of shape:
    `(N°_measurement_points, N°_RamanShifts)`
- **"sigma"** is a 1D numpy array containing all the ramans shift values
    Its' length is `N°_RamanShifts`
- **"params"** is a dictionnary containing measurement parameters
- **"map_params"** is dictionnary containing map parameters
- **"origins"** is a pandas dataframe giving detail on each point in the map scan
    (time of measurement, _coordinates and some other info).

> _Note: It should be noted that the timestamp
    recorded in the origins dataframe is in the Windows 64bit format,
    if you want to convert it to the human readable format,
    you can use the imported "convert_time" function_
'''

see_all_spectra = NavigationButtons(sigma, spectra, autoscale_y=True)

#%%
# put the retreived number of measurements in a variable
# with a shorter name, as it will be used quite often:
try:
    _n_points = int(params['Count'])
except (NameError, KeyError):
    _n_points = len(spectra)
try:
    if params['MeasurementType'] == 'Map':
        # Finding in what axes the scan was taken:
        _x_index, _y_index = np.where(map_params['NbSteps'] > 1)[0]
except (NameError, KeyError):
    _x_index, _y_index = 0, 1

try:
    if params['MeasurementType'] == 'Map':
        # ATTENTION : from this point on in the script,
        # the two relevant dimensions  will be called
        # X and Y regardless if one of them is Z in reality (for slices)
        _n_x, _n_y = map_params['NbSteps'][[_x_index, _y_index]]
except (NameError, KeyError):
    while True:
        _n_x = int(input("Enter the total number of measurement points along x-axis: "))
        _n_y = int(input("Enter the total number of measurement points along y-axis: "))
        if _n_x*_n_y == _n_points:
            print("That looks ok.")
            break
        elif _n_x * _n_y != _n_points:
            warn("\nWrong number of points. Try again:")
            continue
        break

try:
    if params['MeasurementType'] == 'Map':
        _s_x, _s_y = map_params['StepSizes'][[_x_index, _y_index]]
except (NameError, KeyError):
    _s_x = int(input("Enter the size of the step along x-axis: "))
    _s_y = int(input("Enter the size of the step along y-axis: "))
    print("ok")

try:
    if params['MeasurementType'] == 'Map':
        if (initialization['NumberOfLinesToSkip_Beggining']
                + initialization['NumberOfLinesToSkip_End']) > _n_y:
            raise SystemExit('You are skiping more lines than present in the scan.\n'
                             'Please revise your initialization parameters')
        _n_yy = _n_y - initialization['NumberOfLinesToSkip_End'] -\
                       initialization['NumberOfLinesToSkip_Beggining']
    else:
        raise SystemExit("Can't yet handle this type of scan")
except:
    pass
# %%
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

# Removing the lines from top and/or bottom of the map
try:
    skip_lines_up = initialization['NumberOfLinesToSkip_End']
except (ValueError, KeyError):
    skip_lines_up = 0
_start_pos = skip_lines_up * _n_x

try:
    skip_lines_down = initialization['NumberOfLinesToSkip_End']
except (ValueError, KeyError):
    skip_lines_down = 0

if skip_lines_down == 0:
    _end_pos = None
else:
    _end_pos = -np.abs(skip_lines_down) * _n_x

spectra_kept = spectra_kept[_start_pos:_end_pos]

# =============================================================================
# ATTENTION: This next line is likely buggy. The columns containing the coor-
# dinates are not necessarily the ones indicated with _x_index ans _y_index!
# =============================================================================
#_coordinates = origins.iloc[_start_pos:_end_pos, [_x_index+1, _y_index+1]]

#%%
# =============================================================================
# Finding the baseline using the asynchronous least squares method
# =============================================================================
if initialization['BaselineCorrection']:
    b_line = baseline_als(spectra_kept, p=0.01, lam=1e8)
else:
    b_line = np.zeros_like(spectra_kept)

# Remove the eventual offsets:
corrected_spectra = spectra_kept - b_line
corrected_spectra -= np.min(corrected_spectra, axis=1)[:, np.newaxis]

# Visualise the baseline correction:
_baseline_stack = np.stack((spectra_kept, b_line, corrected_spectra), axis=-1)
labels = ['original spectra', 'baseline', 'baseline corrected spectra']
check_baseline = NavigationButtons(sigma_kept, _baseline_stack,
                                   autoscale_y=True, label=labels)

# %%
# =============================================================================
#               Finding the Cosmic Rays with nearest neghbour
#                 and correcting them with median filter...
# =============================================================================
if initialization['CosmicRayCorrection']:
    clf = LocalOutlierFactor(n_neighbors=5, n_jobs=-1)
    prd = clf.fit_predict(corrected_spectra)
    CR_cand_ind = np.where(prd==-1)[0]
else:
    CR_cand_ind = np.asarray([])

if len(CR_cand_ind) > 0:
    # Find the median value for each spectra, but only with regard to
    # its' neighbours from the same line
    med_spectra_x = rolling_median(
                        corrected_spectra.reshape(_n_yy, _n_x, len(sigma_kept)),
                        w_size=5, ax=1,
                        mode='mirror').reshape((-1, len(sigma_kept)))

    titles = [f"candidate from Nearest Neighbour\noriginal spectra N°{i} "
              for i in np.nditer(CR_cand_ind)]
    _ss = np.stack((spectra_kept[CR_cand_ind],
                    corrected_spectra[CR_cand_ind],
                    med_spectra_x[CR_cand_ind]), axis=-1)
    NavigationButtons(sigma_kept, _ss, autoscale_y=True, title=titles,
                      label=['original',
                             'baseline corrected',
                             'median correction of CR']);

    # Apply the correction:
    # (just replace the whole spectra containing the cosmic ray
    # with the median spectra of its' neighborhood)
    if len(CR_cand_ind) > 0:
        corrected_spectra[CR_cand_ind] = med_spectra_x[CR_cand_ind]
# %%
# =============================================================================
#                                     PCA...
# =============================================================================
pca = decomposition.PCA(n_components=initialization['PCA_components'])
pca_fit = pca.fit(corrected_spectra)

spectra_reduced = pca_fit.transform(corrected_spectra)
spectra_denoised = pca_fit.inverse_transform(spectra_reduced)

# =============================================================================
#                  showing the smoothed spectra
# =============================================================================

_s = np.stack((corrected_spectra, spectra_denoised), axis=-1)
see_all_spectra = NavigationButtons(sigma_kept, _s, autoscale_y=True,
                                    label=["corrected spectra", "pca denoised"],
                                    figsize=(12, 12))

# %%
# =============================================================================
#                                   NMF step
# =============================================================================

spectra_cleaned = clean(sigma_kept, spectra_denoised, mode='area')

_n_components = initialization['NMF_NumberOfComponents']
nmf_model = decomposition.NMF(n_components=_n_components, init='nndsvda',
                              max_iter=7, l1_ratio=1)
_start = time()
print('starting nmf... (be patient, this may take some time...)')
mix = nmf_model.fit_transform(spectra_cleaned)
components = nmf_model.components_
reconstructed_spectra1 = nmf_model.inverse_transform(mix)
_end = time()
print(f'nmf done in {_end - _start:.2f}s')

# %%
# =============================================================================
#                    preparing the mixture coefficients
# =============================================================================

mix.resize((_n_x*_n_y), _n_components, )

mix = np.roll(mix, _start_pos, axis=0)
_comp_area = np.empty(_n_components)
for _z in range(_n_components):
    # area beneath each component:
    _comp_area[_z] = np.trapz(components[_z])
    components[_z] /= _comp_area[_z]  # normalizing the components by area
    # renormalizing the mixture coefficients:
    mix[:, _z] *= _comp_area[np.newaxis, _z]
spectra_reconstructed = np.dot(mix, components)
_mix_reshaped = mix.reshape(_n_y, _n_x, _n_components)


# %%
# =============================================================================
#                    Plotting the components....
# =============================================================================
sns.set()  # to make plots pretty :)

# to keep always the same colors for the same components:
col_norm = colors.Normalize(vmin=0, vmax=_n_components)
color_set = ScalarMappable(norm=col_norm, cmap="brg")

# infer the number of subplots and their disposition from n_components
fi, _ax = plt.subplots(int(np.floor(np.sqrt(_n_components))),
                       int(np.ceil(_n_components /
                                   np.floor(np.sqrt(_n_components))
                                   )))
if _n_components > 1:
    _ax = _ax.ravel()
else:
    _ax = [_ax]
for _i in range(_n_components):
    _ax[_i].plot(sigma_kept, components[_i].T, color=color_set.to_rgba(_i))
    _ax[_i].set_title(f'Component {_i}')
    _ax[_i].set_yticks([])
try:
    fi.text(0.5, 0.04,
            f"{params['XlistDataType']} recordings"
            f"in {params['XlistDataUnits']} units",
            ha='center')
except:
    pass

# %%
# =============================================================================
#                       Plotting the main plot...
# =============================================================================
_n_fig_rows = int(np.floor(np.sqrt(_n_components)))
_n_fig_cols = int(np.ceil(_n_components / np.floor(np.sqrt(_n_components))))
fig, _ax = plt.subplots(_n_fig_rows, _n_fig_cols,
                        sharex=True, sharey=True)
if _n_components > 1:
    _ax = _ax.ravel()
else:
    _ax = [_ax]

#%%
def onclick(event):
    '''Double-clicking on a pixel will pop-up the (cleaned) spectrum
    corresponding to that pixel, as well as its deconvolution on the components
    and again the reconstruction for visual comparison'''
    if event.inaxes:
        x_pos = int(np.floor(event.xdata))
        y_pos = int(np.floor(event.ydata))
        broj = int(y_pos*_n_x + x_pos)
        spec_num = int(y_pos*_n_x - _start_pos + x_pos)

        if event.dblclick:
            ff, aa = plt.subplots()
            aa.scatter(sigma_kept, spectra_cleaned[spec_num], alpha=0.3,
                       label=f'(cleaned) spectrum n°{broj}')
            aa.plot(sigma_kept, spectra_reconstructed[broj], '--k',
                    label='reconstructed spectrum')
            for k in range(_n_components):
                aa.plot(sigma_kept, components[k]*mix[broj][k],
                        color=color_set.to_rgba(k),
                        label=f'Component {k} contribution'
                              f'({mix[broj][k]*100:.1f}%)')

# This next part is to reorganize the order of labels,
# so to put the scatter plot first
            handles, labels = aa.get_legend_handles_labels()
            order = list(np.arange(_n_components+2))
            new_order = [order[-1]]+order[:-1]
            aa.legend([handles[idx] for idx in new_order],
                      [labels[idx] for idx in new_order])
            aa.set_title(f'deconvolution of the spectrum from: '
                         f'line {y_pos} & column {x_pos}')
            ff.show()
    else:
        print("you clicked outside the canvas, you bastard :)")


_xcolumn_name = ['X', 'Y', 'Z'][_x_index]
_ycolumn_name = ['X', 'Y', 'Z'][_y_index]

#################################################################################
############## This formatting should be adapted case by case ###################
try:
    _y_ticks = [str(int(x/1000))+'mm' for x in
                np.asarray(origins[_ycolumn_name].iloc[:_n_x*_n_y:_n_x])]
    _x_ticks = [str(int(x/1000))+'mm' for x in
                np.asarray(origins[_xcolumn_name].iloc[:_n_x])]
except:
    pass
#################################################################################
if initialization['AbsoluteScale'] == True:
    scaling = {'vmin': 0, 'vmax': 1}
else:
    scaling={}
for _i in range(_n_components):
    sns.heatmap(_mix_reshaped[:, :, _i], ax=_ax[_i], cmap="jet", annot=False, **scaling)
#    _ax[_i].set_aspect(_s_y/_s_x)
    _ax[_i].set_title(f'Component {_i}', color=color_set.to_rgba(_i),
                      fontweight='extra bold')
#    _ax[_i].set_xticks(10*np.arange(max(12, np.ceil(_n_x/10))))
#    _ax[_i].set_yticks(10*np.arange(max(12, np.ceil(_n_y/10))))
#    print(_i)
    #plt.yticks(10*np.arange(np.floor(_n_y/10)), _y_ticks[::10])
#    try:
#        _ax[_i].set_xticklabels(_x_ticks[::10], size=8, va='bottom')
#        _ax[_i].set_yticklabels(_y_ticks[::10], size=8)
#    except: pass
    plt.setp(_ax[_i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#try:
#    fig.text(0.5, 0.014,
#             f"{origins[_xcolumn_name].columns.to_frame().iloc[0,0]}",
#             #f" in {origins[_xcolumn_name].columns.to_frame().iloc[0,1]}",
#             ha='center')
#    fig.text(0.04, 0.5,
#             f"{origins[_ycolumn_name].columns.to_frame().iloc[0,0]}",
#             #f" in {origins[_ycolumn_name].columns.to_frame().iloc[0,1]}",
#             rotation=90, va='center')
#except: pass
fig.suptitle('Heatmaps showing the abundance of individual components'
             ' throughout the scanned area.')
fig.canvas.mpl_connect('button_press_event', onclick)

# %%
# =============================================================================
#        saving some data for usage in other software (Origin, Excel..)
# =============================================================================
_basic_mix = pd.DataFrame(
        np.copy(mix),
        columns=[f"mixing coeff. for the component {l}"
                 for l in np.arange(mix.shape[1])]
        )
_save_filename_extension = (f"_{_n_components}NMFcomponents_from"
                            f".csv")
_save_filename_folder = '/'.join(x for x in filename.split('/')[:-1])+'/'\
                        + filename.split('/')[-1][:-4]+'/'
if not os.path.exists(_save_filename_folder):
    os.mkdir(_save_filename_folder)

_basic_mix.to_csv(
        f"{_save_filename_folder}MixingCoeffs{_save_filename_extension}",
        sep=';', index=False)
_save_components = pd.DataFrame(
        components.T, index=sigma_kept,
        columns=[f"Component{_i}" for _i in np.arange(_n_components)])
_save_components.index.name = 'Raman shift in cm-1'
_save_components.to_csv(
        f"{_save_filename_folder}Components{_save_filename_extension}",
        sep=';')
# %%
pca_err = np.sum(np.abs(corrected_spectra - reconstructed_spectra1), axis=1)
pca_err.resize(_n_y, _n_x)
plt.figure()
sns.heatmap(pca_err)
plt.show()
plt.title("Checking the reconstruction error from NMF")
