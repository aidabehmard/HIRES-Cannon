# -*- coding: utf-8 -*-
"""
@author: behmardaida, 10/11/2018

- Resolution degradation test
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d

import specmatchemp
import specmatchemp.library
import TheCannon as tc
from modules.normalize import gauss_norm
from modules.cannon_input import ivar_vsini
from modules.synthetic import autocorr


# loads SpecMatch-Emp library
lib_path = '/Users/abehmard/.specmatchemp/library.h5' # path to library file
wavlim = [4990, 6094.9099]                            # specmatch-emp range up to 12th order
lib = specmatchemp.library.read_hdf(lib_path,wavlim=wavlim)

cool = lib.library_params.query('Teff < 5200 and radius < 1')
bad_stars = ['GL570B']
cool = cool[~cool['cps_name'].isin(bad_stars)]


# projected rot. velocity (km/s) corresponding to desired (degraded) resolution
c = 3e5
v_pris = c/(2.3*60000)

R = int(input("Enter desired R: ")) 
v_degrad = c/(2.3*R)
sigma = np.sqrt(v_degrad**2 - v_pris**2)


# rebinning function
# necessary to simulate true sampling of degraded R
reduc = int(60000/R)
def rebin(spectrum):
        bin_flux=[]
        arr = range(0,len(spectrum))
        for i in arr[0::reduc]:
                flux = np.mean(spectrum[i:i+reduc])
                bin_flux.append(flux)
        return bin_flux


# --- cool star subset w snr > 160, one for each bin (width = 100 K) in Teff space
# validation set indices
idx_validate =[0,34,57,70,78,232,237,242,243,247,
        256,257,258,261,267,269,331,341,397,400]

# training set indices
idx_train = cool.lib_index.drop(labels=idx_validate)

spectra_tr = lib.library_spectra[idx_train,0]
spectra_val = lib.library_spectra[idx_validate,0]
err_tr = lib.library_spectra[idx_train,1]
err_val = lib.library_spectra[idx_validate,1]

# ----------------------------------------------------------------------------------
# ----- create training set label matrix  ------------------------------------------
# ----------------------------------------------------------------------------------
rstar = np.asarray(lib.library_params.radius[idx_train].values)
teff = np.asarray(lib.library_params.Teff[idx_train].values)
feh = np.asarray(lib.library_params.feh[idx_train].values)


vsini=[]
for spectrum in spectra_tr:
        # assigns each star a vsini value determined from autocorrelation peak
        vsini_star = autocorr(lib.wav,spectrum,0)
        vsini.append(vsini_star)
        print("done")

vsini=np.asarray(vsini)
labels = np.vstack((teff,rstar,feh,vsini)).T

# pickle things
pickling_on = open('cannon2_specmatch_labels_cntrl_good' + str(R) + '.pkl', "wb")
pickle.dump(labels,pickling_on)




# ---- normalizes spectra via Gaussian filter -----------------
# training set
spectra_tr_norm = []
for spectrum1,val1 in zip(spectra_tr,err_tr):
        spectrum1,val1 = gauss_norm(spectrum1,lib.wav,val1,3)
        spectra_tr_norm.append(spectrum1)

spectra_tr_norm = np.asarray(spectra_tr_norm)

# validation set
spectra_val_norm = []
for spectrum2,val2 in zip(spectra_val,err_val):
        spectrum2,val2 = gauss_norm(spectrum2,lib.wav,val2,3)
        spectra_val_norm.append(spectrum2)

spectra_val_norm = np.asarray(spectra_val_norm)



# --- remove Na doublet (not a good feature to fit - activity-driven)
na_min = 5887
na_max = 5899
na_doublt = np.where((lib.wav > na_min) & (lib.wav < na_max))
lib.wav = np.delete(lib.wav,na_doublt)

# training set
spectra_tr_noNa=[]
err_tr_noNa=[]
for spectrum1,err1 in zip(spectra_tr_norm,err_tr):

        spectrum1 = np.delete(spectrum1,na_doublt)
        err1 = np.delete(err1,na_doublt)
        spectra_tr_noNa.append(spectrum1)
        err_tr_noNa.append(err1)

spectra_tr_noNa=np.asarray(spectra_tr_noNa)
err_tr_noNa=np.asarray(err_tr_noNa)

# validation set 
spectra_val_noNa=[]
err_val_noNa=[]
for spectrum2,err2 in zip(spectra_val_norm,err_val):

        spectrum2 = np.delete(spectrum2,na_doublt)
        err2 = np.delete(err2,na_doublt)
        spectra_val_noNa.append(spectrum2)
        err_val_noNa.append(err2)

spectra_val_noNa=np.asarray(spectra_val_noNa)
err_val_noNa=np.asarray(err_val_noNa)

# --- Rebinning -----------------------------------------------
# training set
spectra_tr_noNa_degrad=[]
err_tr_noNa_degrad=[]
for spectrum1,err1 in zip(spectra_tr_noNa,err_tr_noNa):
        spectrum1 = gaussian_filter1d(spectrum1,sigma,mode='constant')
        spectrum1_rebin = rebin(spectrum1)
        spectra_tr_noNa_degrad.append(spectrum1_rebin)

        err1_rebin = rebin(err1)
        err_tr_noNa_degrad.append(err1_rebin)

spectra_tr_noNa_degrad = np.asarray(spectra_tr_noNa_degrad)
err_tr_noNa_degrad = np.asarray(err_tr_noNa_degrad)


# validation set
spectra_val_noNa_degrad=[]
err_val_noNa_degrad=[]
for spectrum2,err2 in zip(spectra_val_noNa,err_val_noNa):
        spectrum2 = gaussian_filter1d(spectrum2,sigma,mode='constant')
        spectrum2_rebin = rebin(spectrum2)
        spectra_val_noNa_degrad.append(spectrum2_rebin)

        err2_rebin = rebin(err2)
        err_val_noNa_degrad.append(err2_rebin)

spectra_val_noNa_degrad = np.asarray(spectra_val_noNa_degrad)
err_val_noNa_degrad = np.asarray(err_val_noNa_degrad)


pickling_on = open('cannon2_specmatch_spectra_res' + str(R) + '.pkl', "wb")
pickle.dump(spectra_val_noNa_degrad,pickling_on)


# ---------------------------------------------------------------------------------------------
# arrays to hold all test labels, validatation target names, model/real spectra
cannon_labels=[]
cannon_spectra=[]
cannon_s2=[]


# let's bootstrap this
for _ in range(20):
        # inverse variance matrices
        ivar_train = ivar_vsini(err_tr_noNa_degrad,flag='train')
        ivar_validate = ivar_vsini(err_val_noNa_degrad,flag='val')

        # ----- run The Cannon ------------
        # constructs CannonModel object using a 3rd order polynomial vectorizer
        label_names = ("TEFF", "RSTAR", "FE_H", "VSINI")
        model = tc.CannonModel(labels, spectra_tr_noNa_degrad, ivar_train,
                vectorizer = tc.vectorizer.PolynomialVectorizer(label_names,3))

        # Training Step
        theta, s2, metadata = model.train(threads=1)
        cannon_s2.append(s2) # instrinsic model variance


        # Test Step
        # cannon-derived labels for training set - check that Cannon can reproduce actual label values
        cannon_labels_tr, cov_tr, metadata_tr = model.test(spectra_tr_noNa_degrad, ivar_train)

        # cannon-derived labels for validation set
        cannon_labels_val, cov_val, metadata_val = model.test(spectra_val_noNa_degrad, ivar_validate)
        cannon_labels.append(cannon_labels_val)

        # get Cannon-derived model spectra
        flux = model(cannon_labels_val)
        cannon_spectra.append(flux)


# cannon labels of validation sets
cannon_labels=np.asarray(cannon_labels)
cannon_labels=np.concatenate(cannon_labels, axis=0)

# intrinsic model variance for each trained model (20 for ~5% validation sets)
cannon_s2=np.asarray(cannon_s2)

# Cannon-derived model spectra of validation sets
cannon_spectra=np.asarray(cannon_spectra)
cannon_spectra=np.concatenate(cannon_spectra, axis=0)

# --------------------------- pickles things --------------------------------------
# Cannon-derived flux
pickling_on = open('cannon2_cannon_spectra_res' + str(R) + '.pkl', "wb")
pickle.dump(cannon_spectra,pickling_on)

# Cannon-derived parameters
pickling_on = open('cannon2_cannon_labels_res' + str(R) + '.pkl', "wb")
pickle.dump(cannon_labels,pickling_on)

# Cannon model intrinsic variance
pickling_on = open('cannon2_s2_res' + str(R) + '.pkl', "wb")
pickle.dump(cannon_s2,pickling_on)

