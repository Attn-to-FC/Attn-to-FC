import pickle
import sys
import os
import math
import traceback
import argparse
import signal
import atexit
import time

import random
import tensorflow as tf
import numpy as np

import networkx as nx

from myutils import prep, drop

prep('loading sequences... ')
seqdata = pickle.load(open('/nfs/projects/attn-to-fc/data/standard_3dfiles_graphast/dataset.pkl', 'rb'))
drop()

#fid=122380

#print(wsmlnodes)
#print(wsmledges)

def idx2tok(nodelist, path):
    out = list()
    for idx in path:
        out.append(nodelist[idx])
    return out

# one way
def getpaths(tt):
    wsmlpaths = dict()
    i = 0
    
    for fid in seqdata['s%s_nodes' % (tt)].keys():
        wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
        wsmledges = seqdata['s%s_edges' % (tt)][fid]

        g = nx.from_numpy_matrix(wsmledges.todense())

        astpaths = nx.all_pairs_shortest_path(g, cutoff=8)
        outpaths = list()

        for astpath in astpaths:
            source = astpath[0]
            
            if len([n for n in g.neighbors(source)]) > 1:
                continue
            
            for path in astpath[1].values():
                if len([n for n in g.neighbors(path[-1])]) > 1:
                    continue
                
                if len(path) > 1 and len(path) <= 8:
                    newpath = idx2tok(wsmlnodes, path)
                    tmp = [0] * (8 - len(newpath))
                    newpath.extend(tmp)
                    outpaths.append(newpath)
                    
            if(len(outpaths) >= 1000):
                break
            
        wsmlpaths[fid] = outpaths

        if i % 500 == 0:
            print('.', end='')
            sys.stdout.flush()
        if i % 500000 == 0:
            astpaths = dict()
            astpaths['s%s_paths' % tt] = wsmlpaths
            pickle.dump(astpaths, open('astpaths-' + str(i) + '-' + tt + '.pkl', 'wb'))
            wsmlpaths = dict()
        i += 1
        
    return wsmlpaths

astpaths = dict()

prep('calculating shortest paths for test set... ')
tt = 'test'
astpaths['s%s_paths' % tt] = getpaths(tt)
drop()

prep('calculating shortest paths for val set... ')
tt = 'val'
astpaths['s%s_paths' % tt] = getpaths(tt)
drop()

prep('dumping astpaths to pickle... ')
pickle.dump(astpaths, open('astpaths-val-test.pkl', 'wb'))
drop()

astpaths = dict()

prep('calculating shortest paths for train set... ')
tt = 'train'
astpaths['s%s_paths' % tt] = getpaths(tt)
drop()

prep('dumping astpaths to pickle... ')
pickle.dump(astpaths, open('astpaths-final-train.pkl', 'wb'))
drop()

# another way
#for fid in seqdata['s%s_nodes' % (tt)].keys():
    #wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
    #wsmledges = seqdata['s%s_edges' % (tt)][fid]

    #g = nx.from_numpy_matrix(wsmledges.todense())

    #outpaths = list()
    #terminals = list()
    
    #for n in g.nodes():
        #if len([q for q in g.neighbors(n)]) == 1:
            #terminals.append(n)
    
    #for s in terminals:
        #for t in terminals:
            #if s == t:
                #continue
            
            #astpath = nx.shortest_path(g, source=s, target=t)

            #if len(astpath) > 1 and len(astpath) <= 8:
                #newpath = idx2tok(wsmlnodes, astpath)
                #tmp = [0] * (8 - len(newpath))
                #newpath.extend(tmp)
                #outpaths.append(newpath)
                
    #wsmlpaths[fid] = outpaths
    
    #if i % 100 == 0:
        #print('.', end='')
        #sys.stdout.flush()
    #i += 1


#for k in wsmlpaths.keys():
    #v = wsmlpaths[k]

    #for path in v:
        #print(path)
        
    #print(len(v))
