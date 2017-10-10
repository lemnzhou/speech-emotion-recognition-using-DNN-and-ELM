#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:01:22 2017

@author: lemn
"""
import MFCC as mfcc
import PITCH_BASED as pitch_based
import numpy as np

# this function caculate the mean and standard variance of features
def get_mean_and_standard_variance(features):
    tmp_feature = np.array(features)
    n = np.shape(tmp_feature)[1]
    means = np.zeros(n)
    std_variances = np.zeros(n)
    for i in range(n):
        a = tmp_feature[:,i]
        means[i] = np.mean(a)
        std_variances[i] = np.std(a)
    return [means,std_variances]
 
#feature normalization
def feature_normalization(features,mean,variance):
    for i in range(len(features)):
        for j in range(len(features[0])):
            features[i][j] = (features[i][j]-mean[j])/variance[j]
    return features

#combine mfcc feature and pitch-based feature
def combine_feature(features_lists):
    '''
    combine features.every column is a frame feature.
    using np.column_stack
    '''
    return np.column_stack(tuple(features_lists))
    
def save_features(features,save_file):
    '''
    this function is to write features into save_file.
    every 25 frames features are knocked together.
    return value:
        a bool value ,indicate whether this process is well accomplished.
    '''
    f=open(save_file,'w')
    for i in range(np.shape(features)[0]):
        line_str = None
        for j in range(-12,13):
            k = i+j
            if k<0:
                k +=12
            elif k>=np.shape(features)[0]:
                k-=12
            if line_str ==None:
                line_str = ' '.join([str(features[k,m]) for m in range(np.shape(features)[1])])
            else:
                line_str = line_str +' '+ ' '.join([str(features[k,m]) for m in range(np.shape(features)[1])])
        line_str +='\n'
        f.write(line_str)
    f.close()

#check those feature with nan or inf,and then prune them
#def check_feature(feature_file):
#    f = open(feature_file)
#    readlines = f.readlines()
#    f.close()
#    for line in readlines:
#        line = line.strip().split(' ')
#        if 'inf' in line or 'nan' in line:
#            return False
#    return True
    
def feature_get(input_files_list,feature_save_list):
    '''
    input the list of names of waves,and the corresponding save file name list.
    we use the features of 1/3 total waves to get the means and standard variance of the total features.
    then wo normalize every feature of each waves.
    '''
    f = open(input_files_list,'r')
    input_audio_files = f.readlines()
    f.close()
    f = open(feature_save_list,'r')
    save_files = f.readlines()
    f.close()
    i = 0
    n = len(input_audio_files)
    j = int(n/3)
    tmp_features1 = none
    tmp_features2 = []
    tmp_savefilename = []
    means = None
    std_variances = None
    for audio_file,save_file in zip(input_audio_files,save_files):
        #print(audio_file)
        #print(save_file)
        audio_file = audio_file.strip()
        save_file = save_file.strip()
        feature1 = mfcc._extractor(audio_file,n_mfcc=13,n_fft=200,hop_length=80)
        feature2 = pitch_based._extractor(audio_file,window_length = 200,hop_length = 80)
        features = combine_feature([feature1,feature2])
        if i<=j:
            if i==0:
                tmp_features1 = features
            else:
                tmp_features1 = np.vstack([tmp_features1,features])
            tmp_features2.append(features)
            tmp_savefilename.append(save_file)
            if i==j:
                #get means and standard variances and store.
                [means,std_variances] = get_mean_and_standard_variance(tmp_features1)
                f = open('~/experiment/code/feature/iemocap/mean_std_variance','w')
                str_line = ' '.join([str(x) for x in means])+'\n'+' '.join([str(x) for x in std_variances]) +'\n'
                f.write(str_line)
                f.close()
                for k in range(len(tmp_features2)):
                    fs = tmp_features2[k]
                    fl = tmp_savefilename[k]
                    fs = feature_noralization(fs,means,std_variances)
                    save_features(fs,fl)
        else:
            features = feature_noralization(features,means,std_variances)
            save_features(features,save_file)
        i = i+1
        #print(i)
        #if not check_feature(save_file):
           # print(save_file)

    
if __name__=='__main__':
    feature_get('/home/lemn/experiment/code/input_list.txt',
                '/home/lemn/experiment/code/save_list.txt')
    
        
