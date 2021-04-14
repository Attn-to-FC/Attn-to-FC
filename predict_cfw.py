import os
import sys
import traceback
import pickle
import argparse
import collections
from keras import metrics
import random
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import multiprocessing
from itertools import product

from multiprocessing import Pool

from timeit import default_timer as timer

from model import create_model
from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word, init_tf
import keras
import keras.backend as K

from custom.graphlayers import OurCustomGraphLayer
from keras_self_attention import SeqSelfAttention

def gendescr_2inp(model, data, comstok, comlen, batchsize, config, refcoms, count, strat='greedy'):
    # right now, only greedy search is supported...
    
    tdats, coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    
    #coms[:,1] = np.array(list(refcoms.values())[count*batchsize:((count+1)*batchsize)])[:,1]
    coms[:,1] = np.array(list(refcoms.values()))[:,1]
    for i in range(2, comlen):
        results = model.predict([tdats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_3inp(model, data, comstok, comlen, batchsize, config, refcoms, count, strat='greedy'):
    # right now, only greedy search is supported...
    
    tdats, coms, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)

    #coms[:,1] = np.array(list(refcoms.values())[count*batchsize:((count+1)*batchsize)])[:,1]
    coms[:,1] = np.array(list(refcoms.values()))[:,1]
    for i in range(2, comlen):
        results = model.predict([tdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_4inp(model, data, comstok, comlen, batchsize, config, refcoms, count, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats, coms, smls = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    smls = np.array(smls)

    #coms[:,1] = np.array(list(refcoms.values())[count*batchsize:((count+1)*batchsize)])[:,1]
    coms[:,1] = np.array(list(refcoms.values()))[:,1]
    for i in range(2, comlen):
        results = model.predict([tdats, sdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_5inp(model, data, comstok, comlen, batchsize, config, refcoms, count, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    #coms[:,1] = np.array(list(refcoms.values())[count*batchsize:((count+1)*batchsize)])[:,1]
    coms[:,1] = np.array(list(refcoms.values()))[:,1]
    for i in range(2, comlen):
        results = model.predict([tdats, sdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_graphast(model, data, comstok, comlen, batchsize, config, refcoms, count, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)
    

    #coms[:,1] = np.array(list(refcoms.values())[count*batchsize:((count+1)*batchsize)])[:,1]
    coms[:,1] = np.array(list(refcoms.values()))[:,1]
    for i in range(2, comlen):
        results = model.predict([tdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_pathast(model, data, comstok, comlen, batchsize, config, refcoms, count, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats, coms, wsmlpaths = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)
    wsmlpaths = np.array(wsmlpaths)

    print('refcoms len:', len(refcoms))
    print('coms len:', coms.shape)
    #coms[:,1] = np.array(list(refcoms.values())[count*batchsize:((count+1)*batchsize)])[:,1]
    coms[:,1] = np.array(list(refcoms.values()))[:,1]
    for i in range(2, comlen):
        if(config['use_sdats']):
            results = model.predict([tdats, sdats, coms, wsmlpaths], batch_size=batchsize)
        else:
            results = model.predict([tdats, coms, wsmlpaths], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_threed(model, data, comstok, comlen, batchsize, config, refcoms, count, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats, coms = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)

    #coms[:,1] = np.array(list(refcoms.values())[count*batchsize:((count+1)*batchsize)])[:,1]
    coms[:,1] = np.array(list(refcoms.values()))[:,1]
    for i in range(2, comlen):
        results = model.predict([tdats, sdats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def load_model_from_weights(modelpath, modeltype, datvocabsize, comvocabsize, smlvocabsize, datlen, comlen, smllen):
    config = dict()
    config['datvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['datlen'] = datlen # length of the data
    config['comlen'] = comlen # comlen sent us in workunits
    config['smlvocabsize'] = smlvocabsize
    config['smllen'] = smllen

    model = create_model(modeltype, config)
    model.load_weights(modelpath)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modelfile', type=str, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/attn-to-fc/data/standard')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/attn-to-fc/data/outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--num-inputs', dest='numinputs', type=int, default=3)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--zero-dats', dest='zerodats', type=str, default='no')
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset.pkl')
    parser.add_argument('--testval', dest='testval', type=str, default='test')

    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modelfile = args.modelfile
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    num_inputs = args.numinputs
    modeltype = args.modeltype
    outfile = args.outfile
    zerodats = args.zerodats
    datfile = args.datfile
    testval = args.testval

    if outfile is None:
        outfile = modelfile.split('/')[-1]

    K.set_floatx(args.dtype)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel

    sys.path.append(dataprep)
    import tokenizer

    prep('loading tokenizers... ')
    tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/%s' % (dataprep, datfile), 'rb'))
    drop()

    print(zerodats)
    if zerodats == 'yes':
        zerodats = True
    else:
        zerodats = False
    print(zerodats)

    if zerodats:
        v = np.zeros(100)
        for key, val in seqdata['dttrain'].items():
            seqdata['dttrain'][key] = v

        for key, val in seqdata['dtval'].items():
            seqdata['dtval'][key] = v
    
        for key, val in seqdata['dttest'].items():
            seqdata['dttest'][key] = v

    allfids = list(seqdata['c'+testval].keys())
    datvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size

    #datlen = len(seqdata['dttest'][list(seqdata['dttest'].keys())[0]])
    comlen = len(seqdata['c'+testval][list(seqdata['c'+testval].keys())[0]])
    #smllen = len(seqdata['stest'][list(seqdata['stest'].keys())[0]])
    refcoms = {}

    prep('loading config... ')
    (modeltype, mid, timestart) = modelfile.split('_')
    (timestart, ext) = timestart.split('.')
    modeltype = modeltype.split('/')[-1]
    config = pickle.load(open(outdir+'/histories/'+modeltype+'_conf_'+timestart+'.pkl', 'rb'))
    num_inputs = config['num_input']
    drop()

    prep('loading model... ')
    model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer, "SeqSelfAttention":SeqSelfAttention})
    print(model.summary())
    drop()

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outfn = outdir+"/predictions/predict-{}-cfw.txt".format(outfile.split('.')[0])
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
 
    prep("computing predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        st = timer()
        refcoms = {}    
        for fid in fid_set:
            refcoms[fid] = seqdata['c'+testval][fid]
            seqdata['c'+testval][fid] = comstart #np.asarray([stk])
            
        bg = batch_gen(seqdata, testval, config, training=False)
        batch = bg.make_batch(fid_set)

        if config['batch_maker'] == 'datsonly':
            batch_results = gendescr_2inp(model, batch, comstok, comlen, batchsize, config, refcoms, c, strat='greedy')
        elif config['batch_maker'] == 'ast':
            batch_results = gendescr_3inp(model, batch, comstok, comlen, batchsize, config, refcoms, c, strat='greedy')
        elif config['batch_maker'] == 'ast_threed':
            batch_results = gendescr_4inp(model, batch, comstok, comlen, batchsize, config, refcoms, c, strat='greedy')
        elif config['batch_maker'] == 'threed':
            batch_results = gendescr_threed(model, batch, comstok, comlen, batchsize, config, refcoms, c, strat='greedy')
        elif config['batch_maker'] == 'graphast':
            badfids = batch[1]
            batch = batch[0]
            for fid in badfids:
                refcoms.pop(fid)
            batch_results = gendescr_graphast(model, batch, comstok, comlen, batchsize, config, refcoms, c, strat='greedy')
        elif config['batch_maker'] == 'graphast_threed':
            badfids = batch[1]
            batch = batch[0]
            for fid in badfids:
                refcoms.pop(fid)
            batch_results = gendescr_5inp(model, batch, comstok, comlen, batchsize, config, refcoms, c, strat='greedy')
        elif config['batch_maker'] == 'pathast_threed':
            badfids=batch[1]
            batch = batch[0]
            for fid in badfids:
                refcoms.pop(fid)
            batch_results = gendescr_pathast(model, batch, comstok, comlen, batchsize, config, refcoms, c, strat='greedy')
        else:
            print('error: invalid batch maker')
            sys.exit()

        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    outf.close()        
    drop()
