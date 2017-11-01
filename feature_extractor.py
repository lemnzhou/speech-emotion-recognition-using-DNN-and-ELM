#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:01:22 2017

@author: lemn
"""
#import pdb
import MFCC as mfcc
import PITCH_BASED as pitch_based
import librosa
import numpy as np
from scipy.fftpack import fft
import math
from sklearn import preprocessing

def get_mean_and_standard_variance(features):
    features = np.array(features)
    scaler = preprocessing.StandardScaler().fit(features)
    means = scaler.mean_
    std_variances = scaler.std_
    f=open('/home/lemn/experiment/code/feature/iemocap/mean_std_variance','w')
    f.write(' '.join([str(x) for x in means])+'\n')
    f.write(' '.join([str(x) for x in std_variances])+'\n')
    f.close()
    return [means,std_variances]
        
def feature_noralization(feature_file_list):
    f=open(feature_file_list)
    readlines = f.readlines()
    f.close()
    n = len(readlines)
    j = int(n/3)
    features = []
    for i in range(j):
        filename = readlines[i].strip()
        f = open(filename)
        feas = f.readlines()
        f.close()
        for fea in feas:
            fea=fea.strip().split(' ')
            a = [np.float64(x) for x in fea]
            features.append(a)
    [means,std_variances] = get_mean_and_standard_variance(features)
    #f = open('/home/lemn/experiment/code/feature/iemocap/mean_std_variance')
    #means = f.readline().strip()
    #means = [np.float64(x) for x in means.split(' ')]
    #std_variances = f.readline().strip()
    #std_variances = [np.float64(x) for x in std_variances.split(' ')]
    for i in range(n):
        filename = readlines[i].strip()
        print(filename)
        f = open(filename)
        feas = f.readlines()
        f.close()
        f=open(filename,'w')
        for fea in feas:
            fea=fea.strip().split(' ')
            fea = [np.float64(x) for x in fea]
            for j in range(len(fea)):
                fea[j] = (fea[j]-means[j])/std_variances[j]
            fea = ' '.join([str(x) for x in fea])+'\n'
            f.write(fea)
        f.close()

def combine_feature(features_lists):
    '''
    combine features.every column is a frame feature.
    using np.column_stack
    '''
    return np.column_stack(tuple(features_lists))

def get_segment_feature(frame_features,marks):
    [N,M] = np.shape(frame_features)
    segment_features = []
    for i in range(len(marks)):
        if marks[i]==0:
            continue
        a =[]
        for k in range(i-12,i+13):
            t=k
            if k<0:
                t=k+12
            elif k>=N:
                t=k-12
            for j in range(M):
                a.append(frame_features[t,j])
        segment_features.append(a)
    return segment_features
    
def save_features(features,save_file):
    '''
    this function is to write features into save_file.
    
    return value:
        a bool value ,indicate whether this process is well accomplished.
    '''
    f=open(save_file,'w')
    for line in features:
        f.write(' '.join([str(x) for x in line])+'\n')
    f.close()
    

def get_segment_energy_marks(filename,window_length=200,hop_length=80,ratio=0.1):
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
    tmp = energys[:12]
    tmp.extend(energys)
    tmp.extend(energys[-12:])
    segment_energys = []
    for i in range(12,len(tmp)-12):
        t = sum(tmp[i-12:i+12])
        segment_energys.append(t)
    sortedId = np.argsort(segment_energys)
    i = int(len(sortedId)*(1-ratio))
    marks =np.zeros(len(segment_energys))
    for j in range(i,len(sortedId)):
        marks[sortedId[j]]=1
    return marks

def feature_get(input_files_list,feature_save_list):
    #feature_extractors = {mfcc._extractor,pitch_based._extractor}
    f = open(input_files_list,'r')
    input_audio_files = f.readlines()
    f.close()
    f = open(feature_save_list,'r')
    save_files = f.readlines()
    f.close()
    i = 0
    for audio_file,save_file in zip(input_audio_files,save_files):
        audio_file = audio_file.strip()
        save_file = save_file.strip()
        marks = get_segment_energy_marks(audio_file)
        feature1 = mfcc._extractor(audio_file,n_mfcc=13,n_fft=200,hop_length=80)
        feature2 = pitch_based._extractor(audio_file,window_length = 200,hop_length = 80)
        features = combine_feature([feature1,feature2])
        features = get_segment_feature(features,marks)
        save_features(features,save_file)
        print(i)
        i = i+1

    
if __name__=='__main__':
    feature_get('/home/lemn/experiment/code/input_list.txt','/home/lemn/experiment/code/save_list.txt')
    feature_noralization('/home/lemn/experiment/code/save_list.txt')
        
