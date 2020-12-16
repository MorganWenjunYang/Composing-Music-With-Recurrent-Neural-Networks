# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:07:48 2020

@author: YWJ97
"""
import tensorflow as tf
import numpy as np
from prep import build_single_input, statematrix_to_midi, build_input_data

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




class music_gen(tf.keras.Model):
    
    def __init__(self):
        
        super(music_gen, self).__init__()
        
        self.time_lstm1 = tf.keras.layers.LSTM(300,return_sequences=True,dropout=0.5)
        self.time_lstm2 = tf.keras.layers.LSTM(300,return_sequences=True,dropout=0.5)

        self.inter1 = tf.keras.layers.TimeDistributed(self.time_lstm1) #(batch,77,127,300)
        self.inter2 = tf.keras.layers.TimeDistributed(self.time_lstm2) #(batch,77,127,300)
       
        
        # the input of note-axis part of model will be 
        # 1) the note-state vector from previous LSTM stack (batch,127,78,300)
        # 2) where the previous note was chosen to be played (batch,127,78,1)
        # 3) where the previous note was chosen to be articulated (batch,127,78,1)
        # that's why we are using padding here and concatenate the 3 together 
        # please see https://www.tensorflow.org/api_docs/python/tf/pad 
        # https://www.tensorflow.org/api_docs/python/tf/concat
        # for reference

        self.note_lstm1 = tf.keras.layers.LSTM(100,return_sequences=True,dropout=0.5)
        self.note_lstm2 = tf.keras.layers.LSTM(50,return_sequences=True,dropout=0.5)

        self.inter3 = tf.keras.layers.TimeDistributed(self.note_lstm1) #(batch,127,78,100)
        self.inter4 = tf.keras.layers.TimeDistributed(self.note_lstm2) #(batch,127,78,50)

        self.outputs1 = tf.keras.layers.Flatten()
        self.outputs2 = tf.keras.layers.Dropout(.5)
        self.outputs3 = tf.keras.layers.Dense(156, activation='sigmoid') #（batch,127,78,2）

        # output the final result, i.e., probability of playing or articulating certain notes
        self.outputs = tf.keras.layers.Reshape((78,2)) #（batch,78,2）

    
    def call(self, inputs):
        
        
        # For why use permute dimensions and use time distributed layers 
        # please refer to https://keras.io/api/layers/recurrent_layers/time_distributed/
        
        inputs = tf.cast(inputs, tf.float32) #(batch, 127, 78, 82)
        time_inputs = inputs[:, :, :, :80]
        state_inputs = inputs[:, :, :, 81:]
 
        inputs_rotate = tf.keras.backend.permute_dimensions(time_inputs,(0,2,1,3)) #(batch,78,127,80)
        
        inter1 = self.inter1(inputs_rotate)
        inter2 = self.inter2(inter1)
        
        inter2_rotate = tf.keras.backend.permute_dimensions(inter2,(0,2,1,3)) #(batch,127,78,300)
        
        paddings = [[0,0],[0,0],[1,0],[0,0]]
        prev_note_state = tf.pad(state_inputs[:,:,:-1,:], paddings, 'CONSTANT', constant_values=0)   # (batch,127,78,2)
        
        inter_input1 = tf.concat((inter2_rotate,prev_note_state),axis=-1) # (batch,127,78,302)
        inter3 = self.inter3(inter_input1) 
        
        inter_input2 = tf.concat((inter3,prev_note_state),axis=-1) #(batch,127,78,102)
        inter4 = self.inter4(inter_input2) 

        outputs = self.outputs1(inter4)
        outputs = self.outputs2(outputs)
        outputs = self.outputs3(outputs)
        outputs = self.outputs(outputs)
        
        return outputs
    
    # Music composition function
    def compose(self, training_data, length, prob, name = "sample"):

        starting_data = build_single_input(training_data) # Randomly select a starting data
        new_input = tf.concat((np.asarray(starting_data[0][:-1]).reshape(1, 127, 78, 80),
                        np.asarray(starting_data[1][:-1]).reshape(1, 127, 78, 2)), axis = -1) # (1, 127, 78, 82)

        output_state = []

        for _ in range(length):

            pred_state = self.predict(new_input) # Predict statematrix (1, 78, 2)
            
            # Assign 1 when predicted probability > prob
            for i in range(pred_state[0].shape[0]):
                for j in range(pred_state[0].shape[1]):
                    if pred_state[0][i][j] > prob:
                        pred_state[0][i][j] = 1
                    else:
                        pred_state[0][i][j] = 0


            output_state.append(pred_state[0])

            # Combine pred_state to test_state
            new_state = np.concatenate((np.asarray(new_input[0, 1:, :, -2:]).reshape(1, 126, 78, 2), 
                                        np.asarray(pred_state).reshape(1, 1, 78, 2)), axis = 1) # (1, 127, 78, 2)

            new_data = np.asarray(build_input_data(new_state[0])).reshape(1, 127, 78, 80) # Starematrix -> Input_data (1, 127, 78, 80)
            new_input = tf.concat([new_data, new_state], axis = -1) # (1, 127, 78, 82)
        
        sample = statematrix_to_midi(output_state, name = name)
        return sample


# custom loss function
# the output of model is the same shape with the sample's state matrix
# that is (time,note(78),state(2))
# the 2 for each time and note denote the probability of the note being played or articulated repectively in the last step
# we use the negative log likelihood to denote the loss, the log function can avoid the numbers being too small

def my_loss(y_true, y_pred):
#     y_pred=np.asarray(y_pred)
#     y_true=np.asarray(y_true)
    loss=-tf.keras.backend.sum(tf.math.log(y_pred*y_true+(1-y_pred)*(1-y_true)+np.spacing(np.float32(1.0)))) # numeric stablity
    return loss


