#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:40:42 2017
The procedure of deriving the MFCC as follows:
    1.Take the Fourier transform of a singal,obtain the spectrum.
    2.Map the powers of the spectrum above onto the mel scale, using triangular overlapping windows.
    3.Take the logs of the powers at each of the mel frequencies.
    4.Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
    5.The MFCCs are the amplitudes of the resulting spectrum.

But this script using librosa to get the mfcc. 
@author: lemn
"""

import librosa
import numpy as np


def _delta_featrue_funciton(features):
    delta = librosa.feature.delta(features)
    delta = np.vstack((features,delta))
    return delta
    
def _extractor(filename,n_mfcc=13,n_fft = 200,hop_length=80):
    y,sr = librosa.load(filename,None)
    M = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)
    M = _delta_featrue_funciton(M)
    return M.T

if __name__ =='__main__':
    filename = '/home/lemn/experiment/data/MASC/001/neutral/utterance/2011.wav'
    y,sr = librosa.load(filename,None)
    m = _extractor(filename)