# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 23:51:54 2020

@author: YWJ97
"""

import midi
import os, random
import pickle

# =============================================================================
# We will stick to the name convention in the original paper
# Here is the Vocab List:
# #
# # stateMatrix: matrix of state, for state definition see below
# # note: 0-77 lower_bound=24; upper_bound=102 
# # part_position(1) = note
# # pitchclass = 1 of 12 half steps CDEFGAB b#
# # part_pitchclass(12): one-hot pitchclass 
# # state: (1,0) (1,1) (0,0) -> denoting holding or repeating a note
# # context: the count of each pitchclass played in last timestep 
# # part_context(12): rearranged context
# # part_prev_vicinity(50):
#
# input for model: part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0] 
# total number of arguments: 1 + 12 + 50 + 12 + 4 + 1 = 80
# for each of the 78 note you have 80 arguments in above structure
# and we only use sequences of 128 timesteps for training
# so the input data form will be 128*78*80
# =============================================================================

# hyperparameter:
    
batch_size = 10

len_of_seq = 128

upper_bound = 102

lower_bound = 24


def midi_to_statematrix:
    '''
    convert midi file to stateMatrix
    
    '''
    pass

def statematrix_to_midi:
    '''
    convert stateMatrix to  midi file
    
    '''
    pass


def get_part_position:
    pass

def get_part_pitchclass:
    pass

def get_part_prev_vicinity:
    pass

def get_part_context:
    pass


def load_data:
    '''
    load midi data by sequence of 128 timesteps
    
    '''     
    pass

def build_input:
    '''
    for LSTM in the paper
    build input to feed into model based on sequence generated by load_data()
    
    '''
    pass

def build_input_batch:
    '''
    for LSTM in the paper
    build input batch to feed into model based on sequence generated by load_data()
    
    '''
    
    pass

def build_performance_input:
    '''
    for imporved model with performance factor included
    build input to feed into model based on sequence generated by load_data()
    
    '''
    pass

def build_performance_input_batch:
    '''
    for imporved model with performance factor included
    build input to feed into model based on sequence generated by load_data()
    
    '''
    pass