# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:43:06 2020

@author: YWJ97
"""

# =============================================================================
# Vocab List
# #
# # note
# # part_position
# # pitchclass
# # part_pitchclass
# # state
# # context
# # part_context
# # part_prev_vicinity

# input for model: part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0]
# =============================================================================


import itertools
from Original.ori_prep import upperBound, lowerBound

def startSentinel():
    def noteSentinel(note):
        position = note
        part_position = [position]
        
        pitchclass = (note + lowerBound) % 12  # 确定CDEFGAB音名
        part_pitchclass = [int(i == pitchclass) for i in range(12)]   # onehot音名
        
        return part_position + part_pitchclass + [0]*66 + [1] # 1+12+66+1 = 80
    return [noteSentinel(note) for note in range(upperBound-lowerBound)] # 78*80

# note 0-77


# helper function
def getOrDefault(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d


def buildContext(state):
    context = [0]*12   # one for each pitch class
    for note, notestate in enumerate(state):
        if notestate[0] == 1: # if the note is played last timestep
            pitchclass = (note + lowerBound) % 12 # 音名
            context[pitchclass] += 1 # 上一步所演奏的音名计数
    return context
## output batchsize*12

    
def buildBeat(time):
    return [2*x-1 for x in [time%2, (time//2)%2, (time//4)%2, (time//8)%2]]
## what exactly is this?

def noteInputForm(note, state, context, beat):
    position = note
    part_position = [position]

    pitchclass = (note + lowerBound) % 12
    part_pitchclass = [int(i == pitchclass) for i in range(12)]
    # Concatenate the note states for the previous vicinity
    part_prev_vicinity = list(itertools.chain.from_iterable((getOrDefault(state, note+i, [0,0]) for i in range(-12, 13))))

    part_context = context[pitchclass:] + context[:pitchclass]

    return part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0]
## input form of the whole model
## size = 1 + 12 + 50 + 12 + 4 + 1 = 80


def noteStateSingleToInputForm(state,time):
    beat = buildBeat(time)
    context = buildContext(state)
    return [noteInputForm(note, state, context, beat) for note in range(len(state))]

def noteStateMatrixToInputForm(statematrix):
    # NOTE: May have to transpose this or transform it in some way to make Theano like it
    #[startSentinel()] + 
    inputform = [ noteStateSingleToInputForm(state,time) for time,state in enumerate(statematrix) ]
    return inputform

# =============================================================================
# for every midi file:
# after feed into note midiToNoteStateMatrix and then noteStarnoteStateMatrixToInputForm
# the input form would be 
# (music_length, 78(notes), 80(arguments))
# because we only extract sequence of 128 timesteps
# so the music_length = 128
# =============================================================================
