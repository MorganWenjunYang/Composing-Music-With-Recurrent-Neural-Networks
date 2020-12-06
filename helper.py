# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 23:25:01 2020

@author: YWJ97
"""

def shape(lst):
    length = len(lst)
    shp = tuple(shape(sub) if isinstance(sub, list) else 0 for sub in lst)
    if any(x != 0 for x in shp):
        return length, shp
    else:
        return length
    
    