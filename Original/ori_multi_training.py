# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:44:07 2020

@author: YWJ97
"""

import os, random
import numpy as np
from Original.ori_prep import *
from Original.ori_data import *
import pickle

import signal

batch_width = 10 # number of sequences in a batch
batch_len = 16*8 # length of each sequence
division_len = 16 # interval between possible start locations

def loadPieces(dirpath):

    pieces = {}

    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid','.MID'):
            continue

        name = fname[:-4]

        outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname)) # timestep*78*2
        if len(outMatrix) < batch_len:
            continue

        pieces[name] = outMatrix
        print ("Loaded {}".format(name))

    return pieces

def getPieceSegment(pieces):
    piece_output = random.choice(list(pieces.values())) # random midi data
    start = random.randrange(0,len(piece_output)-batch_len,division_len) 
    # extract a sequence with length=16*8=128
    # print "Range is {} {} {} -> {}".format(0,len(piece_output)-batch_len,division_len, start)

    seg_out = piece_output[start:start+batch_len] # 128*78*2 
    seg_in = noteStateMatrixToInputForm(seg_out) # key line 128*78*80

    return seg_in, seg_out

def getPieceBatch(pieces):
    i,o = zip(*[getPieceSegment(pieces) for _ in range(batch_width)]) #10*[128*78*80]
    return np.array(i), np.array(o)

def trainPiece(model,pieces,epochs,start=0):
    stopflag = [False]
    def signal_handler(signame, sf):
        stopflag[0] = True
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(start,start+epochs):
        if stopflag[0]:
            break
        error = model.update_fun(*getPieceBatch(pieces))
        if i % 100 == 0:
            print ("epoch {}, error={}".format(i,error))
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            xIpt, xOpt = map(np.array, getPieceSegment(pieces))
            noteStateMatrixToMidi(np.concatenate((np.expand_dims(xOpt[0], 0), model.predict_fun(batch_len, 1, xIpt[0])), axis=0),'output/sample{}'.format(i))
            pickle.dump(model.learned_config,open('output/params{}.p'.format(i), 'wb'))
    signal.signal(signal.SIGINT, old_handler)
