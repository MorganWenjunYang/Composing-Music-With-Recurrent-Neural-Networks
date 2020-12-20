# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:45:51 2020

@author: YWJ97
"""

### visualize MIDI 2

import midi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# mid_ori =midi.read_midifile('E:/music/midifile/alb_se4.mid')
# mid= midi.read_midifile('E:/music/sample_prob.mid') 
# seq=serialize_midi(mid)
# df=sequence_to_df(seq)
def serialize_midi(mid):
    track=[]
     
    global_time=0
    
    ntrack=len(mid) # number of track in the midi file
    
    tick_till_next = [0 for i in range(ntrack)] # number of tick until next event on each track
    
    evt_each_track = [0 for i in range(ntrack)] # the current processing event in each track
    
    while True:
        
        for i in range(ntrack):
            
            while tick_till_next[i] == 0: # one track may contain two note at the same time
            
                cur_track = mid[i]
                
                cur_track_evt = cur_track[evt_each_track[i]]
                
                if isinstance(cur_track_evt, midi.NoteOnEvent):
                    if cur_track_evt.data[1]!=0:
                        track.append(cur_track_evt)
                else:
                    pass
                
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
        
    return track


def sequence_to_df(sequence):
    tick=[]
    note=[]
    velocity=[]
    for element in sequence:
        tick.append(element.tick)
        note.append(element.data[0])
        velocity.append(element.data[1])
    dict = {'tick': tick, 'note': note, 'velocity': velocity}  
    df = pd.DataFrame(dict) 
    df['time_elapsed'] = df.tick.cumsum()
    return df



def visualize(df):
    sns.set_style('white')
    g = sns.jointplot(df.time_elapsed, df.note,color='k',
        kind='hex', xlim=(min(df.time_elapsed),max(df.time_elapsed)),
        ylim=(16,113),
        joint_kws=dict(gridsize=88))
    g.fig.set_figwidth(30)
    g.fig.set_figheight(15)
    
    sns.despine(left=True, bottom=True) #Remove x and y axes
    plt.setp(g.ax_marg_x, visible=False) #Remove marginal plot of x
    plt.setp(g.ax_marg_y, visible=False) #Remove marginal plot of y
    g.set_axis_labels('', '') #Remove axis labels
    plt.setp(g.ax_joint.get_xticklabels(), visible=False) #Remove x-axis ticks
    plt.setp(g.ax_joint.get_yticklabels(), visible=False) #Remove y-axis ticks
    plt.show()

def visualize_midi(mid):
    seq=serialize_midi(mid)
    df=sequence_to_df(seq)
    visualize(df)


# df_dup=df_ori[df_ori.duplicated(subset=['note','time_elapsed'], keep=False)]
