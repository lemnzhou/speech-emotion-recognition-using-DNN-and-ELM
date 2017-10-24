#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:16:34 2017
IN This feature extractor,we using Hanming window before FFT,and we
chose the hanming window-length the same as the frame window length
what pitch-based feature contains:
pitch-period and harmonics-to-noise ratio.
pitch-period is inversion of pitch(HZ)
harmonics-to-noise is defined as follow:
HNR(m) = 10*log(ACF(x)/(ACF(0)-ACF(x))
m:  this singal, is a frame singal here.
x:  at pitch-period time,in this frame signal.
0:  at zero time,meaning the start time of this frame singal. 
@author: lemn
"""

import numpy as np
import librosa
from scipy.fftpack import fft,ifft

def _autocorrelation(x):
    '''
    get autocorrelation.
    '''
    x = x-np.mean(x)
    norm=np.sum(x**2)
    correlation = np.correlate(x,x,mode='ful')/norm
    t = int(len(correlation)/2)
    return correlation[t:]

def HNR(x,pitch):
    '''this fuction is to caculate harmonics-to-noise ratio,using the autocorrelation function.
    '''
    acf = _autocorrelation(x)
    t0 = acf[0]
    t1 = acf[pitch]
    
    return 10*np.log(np.abs(t1/(t0-t1)))

def _cepstrum(y,win_length,hop_length):
    '''
    the process of caculating cepstrum:
        singal->fft->abs->log->ifft->cepstrum
    This script using the fft and ifft of scipy package.
    y: the singal.
    '''
    # windowing,using hanming window.
    window = np.hamming(win_length)
    for i in range(int(np.floor(y.size/hop_length))):
        win_start = i*hop_length
        win_end = (i+1)*hop_length
        if win_end >=y.size:
            win_end = y.size
        for j in range(win_length):
            if j+win_start<win_end:
                y[win_start+j]=y[win_start+j]*window[j]

    #fft->abs->ifft
    ceps = ifft(np.log(np.abs(fft(y))))
    ceps = ceps.real
    return ceps
    
def _pitch_period_detection(c,fs):
    '''
        this function using the cepstrum(c) of a singal to get the pitch,
        which is the inverse of pitch period. 
    '''
    ms2=int(np.floor(fs*0.002))
    ms20=int(np.floor(fs*0.02))
    t_c = np.array(c[ms2:ms20])
    t_c = t_c.argsort()
    idx =t_c[-1]
    f0 = fs/(ms2+idx-1)
    return 1.0/f0

    
def _delta_featrue_funciton(features):
    delta = librosa.feature.delta(features)
    delta = np.vstack((features,delta))
    return delta


def highest_energy_marks(filename,window_length,hop_length,ratio = 0.2):
    y,fs = librosa.load(filename,sr=None)
    librosa.util.valid_audio(y)
    y = np.pad(y, int(window_length // 2), mode='reflect')  
    y_frames = librosa.util.frame(y,frame_length=window_length,hop_length=hop_length)
    energys = []
    y_frames = y_frames.T
    for frame in y_frames:
        a = fft(frame)
        b = np.abs(a)
        energys.append(np.sum(b))
    sortedId = np.argsort(energys)
    i = int(len(sortedId)*(1-ratio))
    marks =np.zeros(len(energys))
    for j in range(i,len(sortedId)):
        marks[sortedId[j]]=1
    return marks
    
def _extractor(filename,window_length,hop_length):
    y,fs = librosa.load(filename,sr=None)
    fs=int(fs)
    pitch_based_features=[]
    
    #these three lines below are what librosa.mfcc use to frame the signal.use the same method to make sure the same frame numbers. 
    librosa.util.valid_audio(y)
    y = np.pad(y, int(window_length // 2), mode='reflect')  
    y_frames = librosa.util.frame(y,frame_length=window_length,hop_length=hop_length)
    
    for samples_in_window in y_frames.T:
        ceps = _cepstrum(samples_in_window,window_length,hop_length)
        pitch_period = _pitch_period_detection(ceps,fs)
        t = int(fs*pitch_period)
        HNR_m = HNR(samples_in_window,t)
        feature = [pitch_period,HNR_m]
        pitch_based_features.append(feature)
    ret_features = _delta_featrue_funciton(np.matrix(pitch_based_features).T)
    return ret_features.T
    
if __name__ == '__main__':
    filename = '/home/lemn/experiment/data/iemocap/wav/Ses01F_impro01_F000.wav'
    window_length = 200
    hop_length = 80
    marks = highest_energy_marks(filename,window_length,hop_length)
    pitch_based_features=_extractor(filename,window_length,hop_length)
