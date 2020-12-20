# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 23:51:54 2020

@author: YWJ97
"""

# =============================================================================
# This file contains the auxiliary functions we will use in data preprocessing 
# model training and post-processing
# =============================================================================

import midi
import os, random
import numpy as np
import pickle
import tensorflow as tf

# =============================================================================
# We will stick to the name convention in the original paper
# Here is the Vocab List:
# #
# # stateMatrix: matrix of state, for state definition see below time*78*2
# # note: 0-77 <-- based on lower_bound=24; upper_bound=102 
# # part_position(1) = note
# # pitchclass = 1 of 12 half steps CDEFGAB b#
# # part_pitchclass(12): one-hot pitchclass 
# # state: (1,0) (1,1) (0,0) -> denoting holding or repeating a note/ play or articulate
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

# =============================================================================
# MIDI glossary:
# # tick: time until next event in midi: TPQN/Resolution is how many ticks can add up to the length of one timestep
# # velocity: the factor controlling performance
# # pitch: the note value 0-127
# # time signature: things like 4/4 4/8
# =============================================================================

# hyperparameter:
    
batch_size = 10 # size of input batch

len_of_seq = 128 # the number of timestep in each training sequence 

upper_bound = 102 # the upper bound of note

lower_bound = 24 # the lower bound of note

# =============================================================================
# Reference: https://github.com/danieldjohnson/biaxial-rnn-music-composition
# part of our data preprocessing code is written based on our understanding 
# of the implementation of the original author. we also provides extensive 
# comments about the function of each of them
# =============================================================================

def midi_to_statematrix(midifile):
    '''
    convert midi file to stateMatrix
    
    a valid stateMatrix would be of the form (#timestep_of_sequence, 78(number of note), 2(state indicator))
    
    for the state indicator (whether it's played last timestep, whether it's articulated last timestep)
    
    (1,1)  the note was played last timestep and is articulated in last timestep
    
    (1,0)  the note was played last timestep but is not articalated in last timestep 
    
    (0,0): the note wasn't played last timestep 
    
    there are a lot of tracks in a midi file, and many of them don't contain the actual note
    
    also notice for classic piano data, we usually have 2 tracks of music corresponding to left and right hand
    
    in this function we switch between each of the track and combine all the notes played at each timestep to create statematrix
     
    let's start from time 0 for each of the track
    '''
    
    mid=midi.read_midifile(midifile)
    
    state_matrix=[]
    
    global_time=0
    
    ntrack=len(mid) # number of track in the midi file
    
    tick_till_next = [0 for i in range(ntrack)] # number of tick until next event on each track
    
    TPQN=mid.resolution # ticks per quarter note (TPQN): how many ticks can add up to the length of one timestep
    
    note_span=upper_bound-lower_bound
    
    state = [[0,0] for i in range(note_span)] # default state for the inital timestep
    
    state_matrix.append(state) 
    # because this state is the same one as the one outside state_matrix
    # when we edit state outside, the one inside state_matrix will also change
    
    evt_each_track = [0 for i in range(ntrack)] # the current processing event in each track
    
    while True:
        
        if global_time % (TPQN/4) == (TPQN/8): # if we should switch to a new state
            # in this case we set a timestep as 1/16 note
            # this is something we could tweak a little bit
            # but for now we just leave it as it is
            
            # follow original paper's idea the new state will by default hold the note from last time step
            oldstate=state
            state=[[oldstate[i][0],0] for i in range(note_span)]
            state_matrix.append(state)
        
        for i in range(ntrack):
            while tick_till_next[i] == 0: # one track may contain two note at the same time
                cur_track = mid[i]
                cur_track_evt = cur_track[evt_each_track[i]]
                
                if isinstance(cur_track_evt, midi.NoteEvent):
                    if (cur_track_evt.pitch < lower_bound) or (cur_track_evt.pitch >= upper_bound):
                        pass
                        # discard note event below lower_bound and above upper_bound
                        
                    elif isinstance(cur_track_evt,midi.NoteOffEvent) or cur_track_evt.velocity == 0:
                        state[cur_track_evt.pitch-lower_bound]=[0,0]
                        # check MIDI turtorial for more info
                        # in this case a current articulated note will be stopped
                        
                    else:
                        state[cur_track_evt.pitch-lower_bound]=[1,1]
                        # event will be recorded
                        
                elif isinstance(cur_track_evt,midi.TimeSignatureEvent):
                    # if the music sample set/change it's time signature in the middle
                    # we would immediately stop the tranformation
                    # in the author's own word: 'We don't want to worry about non-4 time signatures. Bail early!'
                    if cur_track_evt.numerator not in (2,4):
                        return state_matrix
                
                if evt_each_track[i]>=len(cur_track)-1:
                    tick_till_next[i] = None # if we have observe all the event in current track
                else:
                    evt_each_track[i] += 1
                    tick_till_next[i] = cur_track[evt_each_track[i]].tick
                    
            if tick_till_next[i] is not None:
                tick_till_next[i] -= 1
                
                
        if all(i is None for i in tick_till_next): # if we have observe all the event in all the tracks
            break
                
        global_time = global_time + 1
    
    return state_matrix


def statematrix_to_midi(statematrix,name='example'):
    '''
    reverse of function midi_to_statematrixtime
    
    convert stateMatrix to midi format and write to midi file
    
    Note: 
    1) we will only have one track when we convert it back
    2) we discard the velocity and the tempo info, because we set them to be constant
    
    '''
    statematrix = np.asarray(statematrix)
    
    track = midi.Track()
    
    note_span=upper_bound-lower_bound
    tick_per_time = 55 
    # because in statematrix_to_midi we have global_time % (TPQN/4) == (TPQN/8)
    # TPQN in this set is generally 480, and the default TQPN by midi package is 220
    # in this case 220/4=55
    
    # we need one container for the state in last timestep
    # and initialize it as all zero
    
    base_time=0 # denote the start time
    
    last=[[0,0] for i in range(note_span)]
    
    for time, cur_state in enumerate(statematrix):

        for note in range(note_span):
            
#             Last: [previous state played, previous state articulated]
#             Cur_state: [current state played, current state articulated]
#             play: whether the note is on
#             articulate: where the note is made
            
#             Condition 1: [0,x] [1,x] new note on
#             Condition 2: [1,x] [0,x] previous note off
#             Condition 3: [1,x] [1,1] new note on and previous note off   
                
            # Condition 2
            if last[note][0]==1 and cur_state[note][0]==0:
                    
                track.append(midi.NoteOffEvent(tick=(time-base_time)*tick_per_time,pitch=lower_bound+note))
                base_time=time
            
            # Condition 3
            elif cur_state[note][0]==1 and cur_state[note][1]==1 and last[note][0]==1:

                track.append(midi.NoteOffEvent(tick=(time-base_time)*tick_per_time,velocity=0,pitch=lower_bound+note))
                track.append(midi.NoteOnEvent(tick=(time-base_time)*tick_per_time,velocity=40,pitch=lower_bound+note)) 
                base_time=time
                
            # Condition 1    
            elif cur_state[note][0]==1 and last[note][0]==0 :
                
                track.append(midi.NoteOnEvent(tick=(time-base_time)*tick_per_time,velocity=40,pitch=lower_bound+note)) 
                base_time=time
                # the model in paper does not include velocity factor, so we just hand-pick one for every note
          
        last=cur_state
        
        # the end
    track.append(midi.EndOfTrackEvent(tick=10))
    
    pattern = midi.Pattern()
    
    pattern.append(track)
    
    res = midi.write_midifile("samples/{}.mid".format(name), pattern)
    
    print("{}.mid saved".format(name))
    
    return pattern

def get_part_position(note):
    
    # [0] in the 80 arguments for each timestep and each note in the training sequence
    
    return [note]


def get_part_pitchclass(note):
    
    # [1:13] in the 80 arguments for each timestep and each note in the training sequence
    # get result like CDEFGAB C# Db
    # then encoded it as a vector of size 12
    
    pitch = (note + lower_bound) % 12
    pitch_class = [0 if i != pitch else 1 for i in range(12) ]    
    return pitch_class


def get_part_context(state):
    
    # [63:75] in the 80 arguments for each timestep and each note in the training sequence
    # get the count of note CDEFGAB C# Db played in the last timestep
    # then encoded it as a vector of size 12
    
    # argument: state --- 78*2 for 1 timestep
    
    part_context=[0]*12
    
    for note, twostate in enumerate(state):
        
        if twostate[0] == 1: # the note was played last timestep
            part_context[(note+lower_bound)%12] +=1
    
    return part_context


def get_part_prev_vicinity(note, state):
    
    # [13:63] in the 80 arguments for each timestep and each note in the training sequence
    # get states for the surrounding octave in each direaction
    # 2 for one note, one for play, one for articulate
    # then encoded it as a vector of size 50
    
    # argument: state --- 78*2 for 1 timestep
    
    part_prev_vicinity=[0]*50
    
    for note_idx,two_state in enumerate(state[note-12:note+13]):
        
        part_prev_vicinity[note_idx*2]=two_state[0]
        part_prev_vicinity[note_idx*2+1]=two_state[1]
 
    return part_prev_vicinity

def get_beat(time):
    
    # [75:79] in the 80 arguments for each timestep and each note in the training sequence
    
    return [2*x-1 for x in [time%2, (time//2)%2, (time//4)%2, (time//8)%2]]

def load_data(data_dir):
    '''
    load midi data in the direactory in the form of sequences of 128 timesteps 
    
    transform midi file to statematrix 
    
    structure: dictionary
    
    '''     
    statemat_dict={}
    
    for mid in os.listdir(data_dir):
        music_name=mid[:-4]
        statematrix=midi_to_statematrix(os.path.join(data_dir,mid))
        # statematrix should follow the structure: (#timestep,78,2)
        
        if len(statematrix) < len_of_seq:
            # if the midi sample is too short,i.e. below required length 128, then we have to discard it
            continue
        
        statemat_dict[music_name]=statematrix
        print('load '+ music_name)
    return statemat_dict


def build_input_data_for_one_timestep(time,state):
    
    '''
    for LSTM in the paper
    
    build input to feed into model based on statematrix generated by load_data()
    
    input for model: part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0] 
    
    total number of arguments: 1 + 12 + 50 + 12 + 4 + 1 = 80
    
    shape: 78*2 --> 78*80

    '''
    out=[]
    
    part_context=get_part_context(state)
    
    beat = get_beat(time)
    
    for note_idx, twostate in enumerate(state):
        
        part_position=get_part_position(note_idx)

        part_pitchclass=get_part_pitchclass(note_idx)

        part_prev_vicinity=get_part_prev_vicinity(note_idx, state)
    
        out.append(part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0] )
    
    return out


def build_input_data(statematrix):
    '''
    for LSTM in the paper
    
    build input to feed into model based on statematrix generated by load_data()
    
    input for model: part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0] 
    
    total number of arguments: 1 + 12 + 50 + 12 + 4 + 1 = 80
    
    input shape: timestep*78*2
    
    output shape: timestep*78*80
    '''
    
    out=[build_input_data_for_one_timestep(time,state) for time, state in enumerate(statematrix)]
    
    return out

def build_single_input(statemat_dict):
    '''
    for LSTM in the paper
    
    build single input to feed into model based on statematrix generated by load_data()

    '''
    rand_piece=random.choice(list(statemat_dict.keys()))
    whole_seq=statemat_dict[rand_piece]
    len_whole_seq=len(whole_seq)
    rand_start_point=random.choice(range(0,len_whole_seq-len_of_seq,32))
    
    # there is an interesting setting in original code
    # set a minimum interval to ensure our training sequences are relatively more different
    
    target=whole_seq[rand_start_point:rand_start_point+len_of_seq] # Only need 127
    
    input_seq=build_input_data(target)
    
    return input_seq, target

# =============================================================================
# for every midi file:
# after feed into load_data() and then nuild_single_input()
# the input form would be 
# (music_length, 78(notes), 80(arguments))
# because we only extract sequence of 128 timesteps
# so the music_length = 128
# =============================================================================

def build_input_batch(statemat_dict):
    '''
    for LSTM in the paper
    
    build input batch to feed into model based on statematrix generated by load_data()
    
    '''
    batch_in_data=[]
    batch_target_data=[]
    
    for i in range(batch_size):
        in_data, out_data=build_single_input(statemat_dict)
        batch_in_data.append(in_data)
        batch_target_data.append(out_data)
    
    batch_in_data=np.asarray(batch_in_data)
    batch_target_data=np.asarray(batch_target_data)
    
    return batch_in_data, batch_target_data


def input_batch_generator(statemat_dict):
    
    # training data generator
    
    while True:
        batch=build_input_batch(statemat_dict)
        yield (batch[0],batch[1])
    

    
### Generate (None, 127, 78, 82) training data
def update_input_batch_generator(statemat_dict):
    
    # training data generator
    
    while True:
        batch=build_input_batch(statemat_dict)
        train = tf.concat([batch[0][:,:-1],batch[1][:,:-1]], axis = -1)
        yield (train, batch[1][:,-1])
        
