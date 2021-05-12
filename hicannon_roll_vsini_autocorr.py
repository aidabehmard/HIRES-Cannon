# -*- coding: utf-8 -*-
"""
@author: behmardaida, 01/11/2018

- normalizes HIRES spectra via Gaussian filter
- imports data from SpecMatch-emp library (see Yee et al. 2017)
- runs The Cannon using a rolling leave m-n out scheme
- dumps the normalized spectra, specmatch-emp parameter values, and Cannon-
derived parameter values into pickles for future use 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from pandas import DataFrame
from scipy import interpolate

import specmatchemp
import specmatchemp.library
import TheCannon as tc
from modules.cannon_input import ivar_vsini


# names of targets in order 
pickle_off = open("../HiCannon/pickled/valid_autocorr/broad_spectra_20kms_fine_autocorr.pkl","rb")
broad_spectra = pickle.load(pickle_off)

pickle_off = open("../HiCannon/pickled/valid_autocorr/labels_20kms_fine_autocorr.pkl","rb")
labels = pickle.load(pickle_off)

# load vsini array (augmented)
pickle_off = open("../HiCannon/pickled/valid_autocorr/vsini_20kms_fine_autocorr.pkl","rb")
vsini = pickle.load(pickle_off)


# get variance data from specmatch-emp library 
lib_path = '/Users/abehmard/.specmatchemp/library.h5' # path to library file                      
wavlim = [4990, 6094.9099]                            # specmatch-emp range up to 12th order
lib = specmatchemp.library.read_hdf(lib_path,wavlim=wavlim)

cool = lib.library_params.query('Teff < 5200 and radius < 1')
bad_stars = ['GL570B']
cool = cool[~cool['cps_name'].isin(bad_stars)]

lib_spectra = lib.library_spectra[cool.lib_index,0]
lib_err = lib.library_spectra[cool.lib_index,1]

arr_rad = np.asarray(lib.library_params.radius[cool.lib_index].values)
arr_teff = np.asarray(lib.library_params.Teff[cool.lib_index].values)
arr_feh = np.asarray(lib.library_params.feh[cool.lib_index].values)

err=[]
for i in range(len(cool.lib_index)):
	idx = np.where((labels[:,0][i]==arr_teff) & (labels[:,1][i]==arr_rad))
	err.append(lib_err[idx])

err=np.asarray(err)
err=err[:,0]

# --- remove Na doublet (not a good feature to fit - activity-driven)
na_min = 5887
na_max = 5899
na_doublt = np.where((lib.wav > na_min) & (lib.wav < na_max))
lib.wav = np.delete(lib.wav,na_doublt)

broad_spectra_noNa=[]
for spectrum in broad_spectra:
	spectrum = np.delete(spectrum,na_doublt)
	broad_spectra_noNa.append(spectrum)

broad_spectra_noNa=np.asarray(broad_spectra_noNa)

err_noNa=[]
for s2 in err:
	s2 = np.delete(s2,na_doublt)
	err_noNa.append(s2)

err_noNa=np.asarray(err_noNa)


# ---------------------------------------------------------------------------------------------
# number of spectra duplicates made for vsini test
n_dup = 21

# augment label and variance matrices to reflect broad spectra matrix
broad_labels=[]
broad_err=[]
for i in range(len(labels[:,0])):
	add_rows = np.tile(labels[i],(n_dup,1))
	broad_labels.append(add_rows)

	add_err = np.tile(err_noNa[i],(n_dup,1))
	broad_err.append(add_err)

broad_labels=np.reshape(np.asarray(broad_labels),(len(broad_spectra_noNa[:,0]),3))
broad_labels=np.c_[broad_labels, vsini] 


broad_err=np.reshape(np.asarray(broad_err),(len(broad_spectra_noNa[:,0]),len(err_noNa[0])))

# modify vsini column in label matrix to reflect broadened spectra
for i in range(n_dup):
        broad_labels[:,3][i::n_dup]+=i

# arrays to hold all test labels, validatation target names, model/real spectra 
cannon_labels=[]
specmatch_spectra=[]
cannon_spectra=[]
cannon_s2=[]

# let's bootstrap this
for i in range(int(len(broad_spectra_noNa[:,0])/n_dup)):

	# validation set
	idx_validate = i*n_dup

	# identify duplicates for each spectrum
	n = np.arange(0,n_dup,1)
	exclude = n_dup*i+n

	bool_array = np.zeros(len(broad_spectra_noNa[:,0]), dtype=bool)
	bool_array[exclude] = True

	# establish training set (everything remaining)
	idx_train = np.where(bool_array==False)

	spectra_train = broad_spectra_noNa[idx_train]
	spectra_validate = broad_spectra_noNa[idx_validate]

	specmatch_spectra.append(spectra_validate)
	
	# ----- input for another Cannon run if needed ---------
	# parameter matrix of training set
	specmatch_labels = broad_labels[idx_train]

	# inverse variance matrices
	ivar_train = ivar_vsini(broad_err[idx_train],flag='train') 
	ivar_validate = ivar_vsini(broad_err[idx_validate],flag='val') 

	# ----- run The Cannon ------------
	# constructs CannonModel object using a 3nd order polynomial vectorizer
	label_names = ("TEFF", "RSTAR", "FE_H", "VSINI")
	model = tc.CannonModel(specmatch_labels, spectra_train, ivar_train,
		vectorizer = tc.vectorizer.PolynomialVectorizer(label_names,3))

	scales, fiducials, theta = (model._scales, model._fiducials, model.theta)
	vectorizer = model.vectorizer

	'''
	# let's add regularization
	model.regularization = 100.0
	# lists model variables
	print(model.vectorizer.human_readable_label_vector)
	# checks if there is regularization
	print(model.regularization)
	# checks if there is censoring
	print(model.censors)
	'''
	
	# Training Step
	theta, s2, metadata = model.train(threads=1)
	cannon_s2.append(s2) # instrinsic model variance 

	'''
	# plots instrinsic model variance 
	fig_scatter = tc.plot.scatter(model)
	fig_scatter.axes[0].set_xlim(0, 3500)
	fig_scatter.savefig("scatter_real.png", dpi=300)
	plt.show()

	
	# plots normalized coefficients and scatter of model as a function of wavelength
	fig_theta = tc.plot.theta(model,
    # show the first 4 terms in the label vector.
    	indices=range(4), xlim=(0, 3500),
    	latex_label_names=[
        	r'T_{eff}',
        	r'R_{*}',
        	r'[Fe/H]',
    	])
	fig_theta.savefig("theta_real.png", dpi=300)
	plt.show()
	'''
	
	# Test Step
	# cannon-derived labels for training set - check that Cannon can reproduce actual label values
	cannon_labels_tr, cov_tr, metadata_tr = model.test(spectra_train, ivar_train)
	
	# cannon-derived labels for validation set
	cannon_labels_val, cov_val, metadata_val = model.test(spectra_validate, ivar_validate)
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
pickling_on = open("cannon2_cannon_spectra_norm_3rdOrd_yesspinup_vsini_20kms_autocorr_noNA.pkl", "wb")
pickle.dump(cannon_spectra,pickling_on)

# Cannon-derived parameters
pickling_on = open("cannon2_cannon_labels_norm_3rdOrd_yesspinup_vsini_20kms_autocorr_noNA.pkl", "wb")
pickle.dump(cannon_labels,pickling_on)

# Cannon model intrinsic variance
pickling_on = open("cannon2_s2_norm_3rdOrd_yesspinup_vsini_20kms_autocorr_noNA.pkl", "wb")
pickle.dump(cannon_s2,pickling_on)
'''