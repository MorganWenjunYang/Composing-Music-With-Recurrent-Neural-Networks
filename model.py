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
t_inputs = tf.keras.Input(shape=(128,78,80))

t_inputs_rotate= tf.keras.backend.permute_dimensions(t_inputs,(0,2,1,3)) #(78,128,80)

t_time_lstm1 = tf.keras.layers.LSTM(300,return_sequences=True)
t_time_lstm2 = tf.keras.layers.LSTM(300,return_sequences=True)

t_inter1 = tf.keras.layers.TimeDistributed(t_time_lstm1)(t_inputs_rotate) #(78,128,80)
t_inter2 = tf.keras.layers.TimeDistributed(t_time_lstm2)(t_inter1) #(78,128,80)

t_inter2_rotate= tf.keras.backend.permute_dimensions(t_inter2,(0,2,1,3)) #(128,78,80)
t_outputs = tf.keras.layers.Dense(2,activation='sigmoid')(t_inter2_rotate) #(128,78,2)

time_model=tf.keras.Model(inputs=t_inputs,outputs=t_outputs)




## note axis only

n_inputs = tf.keras.Input(shape=(128,78,80))

n_note_lstm1 = tf.keras.layers.LSTM(100,return_sequences=True)
n_note_lstm2 = tf.keras.layers.LSTM(50,return_sequences=True)

n_inter3 = tf.keras.layers.TimeDistributed(n_note_lstm1)(n_inputs)
n_inter4 = tf.keras.layers.TimeDistributed(n_note_lstm2)(n_inter3)

n_outputs = tf.keras.layers.Dense(2,activation='sigmoid')(n_inter4)

note_model=tf.keras.Model(inputs=n_inputs,outputs=n_outputs)





## biaxial model
  
inputs = tf.keras.Input(shape=(128,78,80))

inputs_rotate= tf.keras.backend.permute_dimensions(inputs,(0,2,1,3)) #(batch,78,128,80)

time_lstm1 = tf.keras.layers.LSTM(300,return_sequences=True,dropout=0.5)
time_lstm2 = tf.keras.layers.LSTM(300,return_sequences=True,dropout=0.5)

inter1 = tf.keras.layers.TimeDistributed(time_lstm1)(inputs_rotate) #(batch,78,128,300)
inter2 = tf.keras.layers.TimeDistributed(time_lstm2)(inter1) #(batch,78,128,300)

note_lstm1 = tf.keras.layers.LSTM(100,return_sequences=True,dropout=0.5)
note_lstm2 = tf.keras.layers.LSTM(50,return_sequences=True,dropout=0.5)

inter2_rotate= tf.keras.backend.permute_dimensions(inter2,(0,2,1,3)) #(batch,128,78,50)

inter3 = tf.keras.layers.TimeDistributed(note_lstm1)(inter2_rotate)
inter4 = tf.keras.layers.TimeDistributed(note_lstm2)(inter3)

outputs = tf.keras.layers.Dense(2,activation='sigmoid')(inter4) #（batch,128,78,2）

model=tf.keras.Model(inputs=inputs,outputs=outputs)


# custom loss function
# the output of model is the same shape with the sample's state matrix
# that is (time,note(78),state(2))
# the 2 for each time and note denote the probability of the note being played or articulated repectively in the last step
# we use the negative log likelihood to denote the loss, the log function can avoid the numbers being too small

def my_loss(y_true, y_pred):
#     y_pred=np.asarray(y_pred)
#     y_true=np.asarray(y_true)
    loss=-tf.keras.backend.mean(tf.math.log(y_pred*y_true+(1-y_pred)*(1-y_true)))
    return loss




def predict_next_step():
    '''
    predict the note in next timestep
    
    '''
    
    pass

def generate_music(feed,length_of_music):
    
    '''
    Generate the music of certain length at your command
    
    '''
    
    pass