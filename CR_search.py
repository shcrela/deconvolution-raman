#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 21:54:49 2021

@author: dejan
"""
import numpy as np
import scipy.ndimage as nd
from utilities import NavigationButtons


#sns.set()
# %%
# =============================================================================
#                                 CR correction...
# =============================================================================
def remove_cosmic_rays(spectra, sigma=None, sensitivity=1):
    '''Attention: The returned spectra is normalized over 1+area
    This method relies on neighbours to identify the cosmic ray candidates
    as parts of spectra too different from the neighbours.

    Parameters:
    ------------------
    spectra: ndarray
        IMPORTANT: if the spectra are comming from the map, then the spectra
        has to be properly shaped.

        >>> spectra.shape
        (nx, ny, n_shifts)

        Otherwise, if the spectra are from temperature series, leave them
        in their shape.

        >>> spectra.shape
        (n_measurements, n_shifts)

    sigma: ndarray
        The Raman shifts. If None, the function will use np.arange(n_shifts)
    sensitivity: float > 0
        higher -> more sensitive (more Cosmic rays will be detected).\n
        smaller -> less sensitive (it will detect only the most prominent CRs).

    Returns:
    -----------------
    cleaned spectra : ndarray
        The array of the same shape as the imput spectra, only normalized and
        (hopefully) with cosmic rays removed.
    '''
    nedim = np.ndim(spectra)
    if nedim == 3:
        ny, nx, n_shifts = spectra.shape
    elif nedim == 2:
        nx = 1
        ny, n_shifts = spectra.shape
    else :
        raise ValueError("Check your input spectra's shape.")
    if sigma is None:
        sigma = np.arange(n_shifts)
    if nx > 3:
        # construct the footprint pointing to the pixels surrounding any given pixel:
        tt = np.zeros((5,nx,1))
        tt[2,:] = tt[:,2] = tt[1,1] = tt[1,3] = tt[3,1] = tt[3,3] = 1
        tt[2,2] = 0
    elif nx == 1:
        tt = np.ones((5,1))
    else:
        raise ValueError(f'The width "nx" should be > 3, and yours is {nx}.')
    kkk = tt.ravel()[:, np.newaxis]
    spectra = spectra.reshape(nx*ny, n_shifts)
    scaling_koeff = 1 + np.abs(np.trapz(spectra, x=sigma,
                                        axis=-1)[:, np.newaxis])
    normalized_spectra = np.copy(spectra / scaling_koeff)
    # each pixel has the median value of its surrounding neighbours:
    median_spectra3 = np.copy(nd.median_filter(normalized_spectra,
                                            footprint=kkk, mode='nearest'))

    # I will only take into account the positive values (CR):
    coarsing_diff = normalized_spectra - median_spectra3

    # For each spectra, find the zone where the difference between the given
    # spectra and its' neighbours is the greatest.
    basic_candidates = np.nonzero(coarsing_diff >\
                                  20*np.std(coarsing_diff) / sensitivity)
    sind = basic_candidates[0] # the spectra containing very bad neighbours
    # rind = basic_candidates[1] # each element from the "very bad neighbour"
    if len(sind) > 0:
        # =====================================================================
        #               We want to extend the "very bad neighbour" label
        #           to ext_size adjecent family members in each such spectra:
        # =====================================================================
        candidates_mask = np.zeros_like(normalized_spectra, dtype=bool)
        candidates_mask[basic_candidates] = True
        largest_cr = 5*np.sum(candidates_mask, axis=-1)
        extended_mask = np.copy(candidates_mask)
        for i, zone in enumerate(candidates_mask):
            if largest_cr[i] > 0:
                extended_mask[i] = nd.binary_dilation(zone,
                                        structure=np.ones((largest_cr[i]),))
    cleaned_spectra = np.copy(normalized_spectra)
    CR_cand_ind = np.unique(sind)

    # If one is willing to neglect the probability of the situation where
    # one spectra has its' cosmic ray at the rightmost pixel, and the spectrum
    # that follows it has the cosmic ray at the leftmost pixel, then we can
    # label the zones to be replaced like this:
    labels = nd.label(extended_mask.ravel(),
                      structure=np.ones(3))[0].reshape(extended_mask.shape)
    # `labels` label each zone to be replaced differently

    # These are the points where the zone to be replaced starts
    # (some spectra may have multiple such zones)
    stiching_points = np.nonzero(np.diff(extended_mask, n=1, axis=-1,
                                         append=0)==1)
    koefs = normalized_spectra[stiching_points]/\
        median_spectra3[stiching_points]

    for i in np.arange(1, len(koefs)+1):
        cleaned_spectra[labels==i] = median_spectra3[labels==i]*koefs[i-1]
    # Ilustration:
    _ss = np.stack((normalized_spectra[CR_cand_ind],
                    cleaned_spectra[CR_cand_ind]),
                   axis=-1)
    title = [f"indice={i}" for i in CR_cand_ind]
    check_CR_candidates = NavigationButtons(sigma, _ss,
                                            autoscale_y=True,
                                            title=title,
                                            label=['normalized spectra',
                                                   'median correction']);
    check_CR_candidates.figr.suptitle("Cosmic Rays Correction")

    return cleaned_spectra.reshape((ny, nx, n_shifts)), CR_cand_ind

