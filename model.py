# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:07:48 2020

@author: YWJ97
"""
import tensorflow as tf

# =============================================================================
# In the section, we will build LSTM model according to the definition stated in 
# the original paper using keras Model API
# =============================================================================
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

# class music_gen(tf.keras.Model):

#   def __init__(self):
#     super(music_gen, self).__init__()
#     self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#     self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
#     self.dropout = tf.keras.layers.Dropout(0.5)

#   def call(self, inputs, training=False):
#     x = self.dense1(inputs)
#     if training:
#       x = self.dropout(x, training=training)
#     return self.dense2(x)


# =============================================================================
# We are going to stack LSTM cell in note axis
# Intuitionally, RNNs are inherently deep in time,
# since their hidden state is a function of all previous hidden states. 
# The question that inspired this was whether RNNs could also benefit from depth in space; 
# that is from stacking multiple recurrent hidden layers on top of each other, 
# just as feedforward layers are stacked in conventional deep networks.
# =============================================================================

## time axis only
inputs = tf.keras.Input(shape=(78,80))
t1 = tf.keras.layers.LSTM(78)(inputs)
t2 = tf.keras.layers.LSTM(78)(t1)
p1 = tf.keras.layers.LSTM(78)(t2)
p2 = tf.keras.layers.LSTM(78)(p1)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(p2)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


## note axis only


## biaxial model
  
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback

model=Sequential()
model.add(LSTM(512,  return_sequences=True, stateful=True,
                     batch_input_shape=(10, 128, 78)))
model.add(Dropout(0.2))
model.add(LSTM(512,  return_sequences=False))
model.add(Dense(78))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def predict_next_step:
    '''
    predict the note in next timestep
    
    '''
    pass
