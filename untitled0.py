#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:01:22 2017

@author: lemn
"""
import MFCC as mfcc
import PITCH_BASED as pitch_based
import numpy as np
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
    tmp_feature = np.array(features)
    n = np.shape(tmp_feature)[1]
    means = np.zeros(n)
    std_variances = np.zeros(n)
    for i in range(n):
        a = tmp_feature[:,i]
        means[i] = np.mean(a)
        std_variances[i] = np.std(a)
    return [means,std_variances]
        
def feature_noralization(features,mean,variance):
    for i in range(len(features)):
        for j in range(len(features[0])):
            features[i][j] = (features[i][j]-mean[j])/variance[j]
    return features

def combine_feature(features_lists):
    '''
    combine features.every column is a frame feature.
    using np.column_stack
    '''
    return np.column_stack(tuple(features_lists))
    
def save_features(features,save_file):
    '''
    this function is to write features into save_file.
    
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
                line_str = line_str + ' '.join([str(features[k,m]) for m in range(np.shape(features)[1])])
        line_str +='\n'
        f.write(line_str)
    f.close()
    
def check_feature(feature_file):
    f = open(feature_file)
    readlines = f.readlines()
    f.close()
    for line in readlines:
        line = line.strip().split(' ')
        if 'inf' in line or 'nan' in line:
            return False
    return True
    
def feature_get(input_files_list,feature_save_list):
    #feature_extractors = {mfcc._extractor,pitch_based._extractor}
    f = open(input_files_list,'r')
    input_audio_files = f.readlines()
    f.close()
    f = open(feature_save_list,'r')
    save_files = f.readlines()
    f.close()
    i = 0
    n = len(input_audio_files)
    j = int(n/3)
    tmp_features1 = []
    tmp_features2 = []
    tmp_savefilename = []
    means = None
    std_variances = None
    for audio_file,save_file in zip(input_audio_files,save_files):
        #print(audio_file)
        #print(save_file)
        audio_file = audio_file.strip()
        save_file = save_file.strip()
#        feature1 = mfcc._extractor(audio_file,n_mfcc=13,n_fft=200,hop_length=80)
#        feature2 = pitch_based._extractor(audio_file,window_length = 200,hop_length = 80)
#        features = combine_feature([feature1,feature2])
        if i<=j:
            feature1 = mfcc._extractor(audio_file,n_mfcc=13,n_fft=200,hop_length=80)
            feature2 = pitch_based._extractor(audio_file,window_length = 200,hop_length = 80)
            features = combine_feature([feature1,feature2])
            tmp_features1.extend(features)
            tmp_features2.append(features)
            tmp_savefilename.append(save_file)
            if i==j:
                [means,std_variances] = get_mean_and_standard_variance(tmp_features1)
                f = open('~/experiment/code/feature/iemocap/mean_std_variance','w')
                str_line = ' '.join([str(x) for x in means])+'\n'+' '.join([str(x) for x in std_variances]) +'\n'
                f.write(str_line)
                f.close()
#                for k in range(len(tmp_features2)):
#                    fs = tmp_features2[k]
#                    fl = tmp_savefilename[k]
#                    fs = feature_noralization(fs,means,std_variances)
#                    save_features(fs,fl)
                fs = tmp_features2[-1]
                fl = tmp_savefilename[-1]
                fs = feature_noralization(fs,means,std_variances)
                save_features(fs,fl)
        #else:
#            features = feature_noralization(features,means,std_variances)
#            save_features(features,save_file)
        i = i+1
        print(i)
        if not check_feature(save_file):
            print(save_file)
        
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
    feature_get('/home/lemn/experiment/code/input_list.txt',
                '/home/lemn/experiment/code/save_list.txt')
    
        