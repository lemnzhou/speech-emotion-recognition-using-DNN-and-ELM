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
#modules = {
#        'mfcc':mfcc,
#        'pitch_based':pitch_based
#        }
#
#def feature_clusters(features_arr):
#    '''
#    via this function to decide which features to be extracted.
#    features include:
#        mfcc
#        pitch_based
#    '''
#    feature_extractors = []
#    if features_arr == None:
#        return feature_extractors
#    for feature_name in features_arr:
#        feature_extractors.append(switch[feature_name])
#    return feature_extractors
#
#def _parse(input_str):
#    '''
#    parsing the input string.using the _parse function in corresponding feature module.
#    return these arguments and corresponding _extractor functions.
#    '''
#    substr = input_str.split().split(',')
#    input_files_list = substr[0].strip()
#    feature_save_list = substr[1].strip()
#    feature_names = []
#    arg_strs = []
#    for s in substr[2:]:
#        [feature_name,arg_str] = s.split().strip(':')
#        arg_str = arg_str.split()
#        feature_name = feature_name.split()
#        
#        module_name = modules[feature_name]
#        feature_extractors.append(module_name._extractor)     #this feature's  _extractor
#        arg_lists.append(module_name._parse(arg_str))   #this feature's parsing function parse it's inputstring
#        
#    return [input_files_list,feature_save_list,feature_extractors,arg_list]
#
#    
#
##def input_check(input_files_list,feature_save_list,features_arr=['mfcc'],features_args):
#    '''
#    this function is to check the inputs of feature_extract function.
#    several points:
#        1.audio_extraction_list and feature_save_list is a true path of a txt file,they have the same counting lines.
#        2.features_arr and the features_args have the same length.
#    return value is also a bool value.
#    '''
#    
#def feature_extract(input_files_list,feature_save_list,features_arr=['mfcc'],features_args):
#    '''
#    using feature_extractor to extract features.
#    audio_extraction_list:
#        those audio files to extract features
#    feature_save_list:
#        those audio files' features files save paths.
#    features_arr:
#        those features that each audio file will extract. e.g. mfcc,pitch_based
#    features_args:
#        correlation to features_arr,are these features extractors' arguments, as string type. 
#    return value is also a bool value.
#    '''
#    feature_extractors = feature_clusters(features_arr)
#    f = open(input_files_list,'r')
#    input_audio_files = f.readlines()
#    f.close()
#    f = open(feature_save_list,'r')
#    save_files = f.readlines()
#    f.close()
#    for audio_file in input_audio_file,save_file in save_files:
#        features = None
#        for _extract_function in feature_extractors,arg_list in features_args:
#            feature = _extract_function(audio_file.strip(),arg_list)
#            features.append(feature)
#            
#        features = combine_feature(features)
#        save_features(features,save_file.strip())
#    return True

def get_mean_and_standard_variance(features):
    #pdb.set_trace()
#    n = len(features[0])
#    means = np.zeros(n)
#    std_variances = np.zeros(n)
#    for i in range(n):
#        a = []
#        for j in range(len(features)):
#            a.append(features[j][i])
#        a = np.array(a)
#        means[i] = np.mean(a)
#        std_variances[i] = math.sqrt(np.va)
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
    #j = int(n/3)
    #features = []
    #for i in range(j):
    #    filename = readlines[i].strip()
    #    f = open(filename)
    #    feas = f.readlines()
    #    f.close()
    #    for fea in feas:
    #        fea=fea.strip().split(' ')
    #        a = [np.float64(x) for x in fea]
    #        features.append(a)
    #[means,std_variances] = get_mean_and_standard_variance(features)
    f = open('/home/lemn/experiment/code/feature/iemocap/mean_std_variance')
    means = f.readline().strip()
    means = [np.float64(x) for x in means.split(' ')]
    std_variances = f.readline().strip()
    std_variances = [np.float64(x) for x in std_variances.split(' ')]
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
    
#def check_feature(features):
#    f = open(feature_file)
#    readlines = f.readlines()
#    f.close()
#    for line in readlines:
#        line = line.strip().split(' ')
#        if 'inf' in line or 'nan' in line:
#            return False
#    return True

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
#    n = len(input_audio_files)
#    j = int(n/3)
#    tmp_features1 =[]
#    tmp_features2 = []
#    tmp_savefilename = []
#    means = None
#    std_variances = None
    for audio_file,save_file in zip(input_audio_files,save_files):
        audio_file = audio_file.strip()
        save_file = save_file.strip()
        marks = get_segment_energy_marks(audio_file)
        feature1 = mfcc._extractor(audio_file,n_mfcc=13,n_fft=200,hop_length=80)
        feature2 = pitch_based._extractor(audio_file,window_length = 200,hop_length = 80)
        features = combine_feature([feature1,feature2])
        features = get_segment_feature(features,marks)
#        if i<=j:
##            if i==0:
##                tmp_features1 = features
##            else:
##                np.vstack(tuple([tmp_features1,features]))
#            tmp_features1.extend(features)
#            tmp_features2.append(features)
#            tmp_savefilename.append(save_file)
#            if i==j:
#                [means,std_variances] = get_mean_and_standard_variance(tmp_features1)
#                f = open('/home/lemn/experiment/code/feature/iemocap/mean_std_variance','w')
#                str_line = ' '.join([str(x) for x in means])+'\n'+' '.join([str(x) for x in std_variances]) +'\n'
#                f.write(str_line)
#                f.close()
#                for k in range(len(tmp_features2)):
#                    fs = tmp_features2[k]
#                    fl = tmp_savefilename[k]
#                    fs = feature_noralization(fs,means,std_variances)
#                    save_features(fs,fl)
#                    print(fl)
#        else:
#            features = feature_noralization(features,means,std_variances)
#            save_features(features,save_file)
        save_features(features,save_file)
        print(i)
        i = i+1
#        if not check_feature(save_file):
#            print(save_file)
#        
#def extractor_run():
#    '''
#    '''
#    input_format_sring=''.join(['please input as this:\n',
#    'input_files_list,\nfeature_save_list,\n',
#    'feature_name1:arg_string1,feature_name2:arg_string2...\n',
#    'the args_string contain argument like this argument_name=argument_value,n=3,for example.\n',
#    'It is no warry if the strings contain some blank spaces.\n'])
#    
#    #input_str = input(input_format_sring)
#    input_str = ''.join([
#    '/home/lemn/experiment/code/Feature_extractor/filenames_list.txt',
#    '/home/lemn/experiment/code/Feature_extractor/savefiles_list.txt',
#    'mfcc:n_mfcc=13,n_fft=200,hop_length=80',
#    'pitch_based:window_length = 200,hop_length = 80'
#    ])
#    window_length=200
#    hop_length=80
#    n_mfcc=13
#    n_fft=200

    
if __name__=='__main__':
    feature_get('/home/lemn/experiment/code/input_list.txt','/home/lemn/experiment/code/save_list.txt')
    feature_noralization('/home/lemn/experiment/code/save_list.txt')
        
