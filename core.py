# -*- coding: utf-8 -*-
"""
@author: behmardaida

- processes training set spectra + labels and test set spectra for The Cannon
- pulls from SpecMatch-Emp library (Yee+2017) of HIRES spectra
- runs The Cannon on the CKS-cool sample

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

import specmatchemp
import specmatchemp.library
import TheCannon as tc
from modules.normalize import gauss_norm
from modules.cannon_input import ivar_vsini
from modules.synthetic import autocorr


# ---------------------------------------------
# ---------- Training Set ---------------------
# ---------------------------------------------

# loads SpecMatch-Emp library to create training set
lib_path = input('Enter path to SpecMatch-Emp library: ') # e.g., '/Users/abehmard/.specmatchemp/library.h5'                     
wavlim = [4990, 6094.9099]                                # wavelength range up to 12th HIRES order
lib = specmatchemp.library.read_hdf(lib_path,wavlim=wavlim)

# subset of SpecMatch-Emp library that suits user purposes (this version = cool stars)
cool = lib.library_params.query('Teff < 5200 and radius < 1')
print("There are "+str(len(lib.library_spectra[cool.lib_index,:,:]))+" targets in this training set") 

bad_stars = [input('Enter names of targets to exclude: ')] # e.g., 'GL570B'
cool = cool[~cool['cps_name'].isin(bad_stars)]
print("Removing unwanted stars, there are now "+str(len(lib.library_spectra[cool.lib_index,:,:]))+" targets in this training set")

# normalizes training spectra via Gaussian filter
for ind in cool.lib_index:
	lib.library_spectra[ind,0],lib.library_spectra[ind,1] = 
		gauss_norm(lib.library_spectra[ind,0],lib.wav,lib.library_spectra[ind,1],3) 

# removes Na doublet - activity-driven feature 
spectra_tr,errs_tr,wavs_tr = rm_Na_doublt(lib.library_spectra[cool.lib_index,0],lib.library_spectra[cool.lib_index,1],lib.wav)

# copies and broadens training set
augment=6         # broaden spectra copies up to augment-1 km/s
kernel_width=51   # ~ +/-25 km/s
cool_size=len(cool.lib_index)

broad_spectra_tr=[]
vsini=[]
for ind in range(cool_size):
	for i in np.arange(0,augment,1):
		broad_spectrum_tr = broaden_smsyn(spectra_tr[ind],kernel_width,1,0,i+1e-10)
		broad_spectra_tr.append(broad_spectrum_tr)

		# assigns each star a vsini value determined from autocorrelation peak
		vsini_star = autocorr(wavs_tr,broad_spectrum_tr,i)
		vsini.append(vsini_star)
	print("assigned vsini")

vsini = np.asarray(vsini)
spectra_tr = np.asarray(broad_spectra_tr)

# constructs training set label matrix
teff = [lib.library_params.Teff[ind] for ind in cool.lib_index]
rstar = [lib.library_params.radius[ind] for ind in cool.lib_index]
feh = [lib.library_params.feh[ind] for ind in cool.lib_index]
labels_tr = np.vstack((teff,rstar,feh)).T

# augment label and spectral errs matrices to reflect broad spectra matrix
broad_labels_tr = [np.tile(labels_tr[i],(augment,1)) for i in range(len(labels_tr[:,0]))]
errs_tr = np.asarray([np.tile(errs_tr[i],(augment,1)) for i in range(len(labels_tr[:,0]))])
errs_tr = np.reshape(errs_tr, (846,-1))  

# reshape training label matrix and append vsini column
broad_labels_tr = np.reshape(np.asarray(broad_labels_tr),(len(broad_spectra_tr[:,0]),3))
labels_tr = np.c_[broad_labels_tr, vsini] 

# ---------------------------------------------
# ---------- Test/Validation Set --------------
# ---------------------------------------------

# loads test/validation set spectra, flux errs, and wavelength grid
pickle_off = open("CKS-cool/cannon_run/CKS_cool_shifted_spectra.pkl","rb")
spectra_val = pickle.load(pickle_off)

pickle_off = open("CKS-cool/cannon_run/CKS_cool_err.pkl","rb")
errs_val = pickle.load(pickle_off)

pickle_off = open("CKS-cool/cannon_run/CKS_cool_wavs.pkl","rb")
wavs_val = pickle.load(pickle_off)


# resample onto lib.wav grid to match training set spectra
spectra_val_resamp=[] 
errs_val_resamp=[] 
for spectrum_val,err_val,wav in zip(spectra_val,errs_val,wavs_val):
	spectrum_val = np.interp(lib.wav,wav,spectrum_val)
	err_val = np.interp(lib.wav,wav,err_val) 

	spectra_val_resamp.append(spectrum_val)
	errs_val_resamp.append(err_val)

spectra_val_resamp = np.asarray(spectra_val_resamp)
errs_val_resamp = np.asarray(errs_val_resamp)

# normalize test/validation set spectra
spectra_val_resamp,errs_val_resamp = 
	[gauss_norm(spectrum_val,lib.wav,err_val) for spectrum_val,err_val in zip(spectra_val_resamp,errs_val_resamp),3]  
       
# removes Na doublet (wavs_val = wavs_tr)
spectra_val,errs_val,wavs_val = rm_Na_doublt(spectra_val_resamp,errs_val_resamp,lib.wav)


# ----------------------------------------------------------------------------------
# ----- run The Cannon  ------------------------------------------------------------
# ----------------------------------------------------------------------------------
# inverse variance matrices
ivar_tr = ivar_vsini(errs_tr,flag='train')
ivar_val = ivar_vsini(errs_val,flag='val')

# ----- run The Cannon ------------
# constructs CannonModel object using a 3rd order polynomial vectorizer
label_names = ("TEFF", "RSTAR", "FE_H", "VSINI")

vectorizer = tc.vectorizer.PolynomialVectorizer(label_names,3)
model = tc.CannonModel(labels_tr, spectra_tr, ivar_tr,
        vectorizer = vectorizer)


'''
# can save the model to a disk
# can shoose to save training set fluxes+inverse variances (or not) -
# may be useful if you want to re-train the model w/ regularization, etc.
model.write = ("cool-stars.model", include_training_set_spectra=True)

# to read saved model from disk
new_model = tc.CannonModel.read("cool-stars.model")

# checks if trained
new_model.is_trained
# >> True
'''


# Training Step
theta, s2, metadata = model.train(threads=1)

# Test Step
# cannon-derived labels for training set 
cannon_labels_tr, cov_tr, metadata_tr = model.test(spectra_tr, ivar_tr)

# cannon-derived labels for validation set
cannon_labels_val, cov_val, metadata_val = model.test(spectra_val, ivar_val)

# get Cannon-derived model spectra
flux = model(cannon_labels_val)


# --------------------------- pickles things --------------------------------------
# cannon labels of validation set
cannon_labels=np.asarray(cannon_labels_val)
cannon_labels=np.concatenate(cannon_labels, axis=0)

# intrinsic model variance for trained model 
cannon_s2=np.asarray(s2)

# Cannon-derived model spectra of validation sets
cannon_spectra=np.asarray(flux)
cannon_spectra=np.concatenate(cannon_spectra, axis=0)

# Cannon-derived flux
pickling_on = open('cannon_cannon_spectra_CKS_cool.pkl', "wb")
pickle.dump(cannon_spectra,pickling_on)

# Cannon-derived parameters
pickling_on = open('cannon_cannon_labels_CKS_cool.pkl', "wb")
pickle.dump(cannon_labels,pickling_on)

# Cannon model intrinsic variance
pickling_on = open('cannon_s2_CKS_cool.pkl', "wb")
pickle.dump(cannon_s2,pickling_on)




