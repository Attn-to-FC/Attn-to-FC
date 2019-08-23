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

def gendescr_2inp(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...
    
    tdats, coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)

    for i in range(1, comlen):
        results = model.predict([tdats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_3inp(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...
    
    tdats, coms, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_4inp(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats, coms, smls = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    smls = np.array(smls)

    #print(sdats)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_5inp(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    #print(sdats)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_graphast(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    #print(sdats)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_threed(model1, model2, data, comstok, comlen, batchsize, config1, config2, strat='greedy'):
    # right now, only greedy search is supported...

    tdats, sdats, coms = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)

    for i in range(1, comlen):
        results1 = model1.predict([tdats, sdats, coms], batch_size=batchsize)
        results2 = model2.predict([tdats, coms], batch_size=batchsize) # TODO currently only works for attendgru
        for c, (s1, s2) in enumerate(zip(results1, results2)):
            coms[c][i] = np.argmax(np.mean([s1, s2], axis=0))

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
    parser.add_argument('modelfile1', type=str, default=None)
    parser.add_argument('modelfile2', type=str, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/funcom/data/standard')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/funcom/data/outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--num-inputs', dest='numinputs', type=int, default=3)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--zero-dats', dest='zerodats', action='store_true', default=False)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--testval', dest='testval', type=str, default='test')

    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    modelfile1 = args.modelfile1
    modelfile2 = args.modelfile2
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    num_inputs = args.numinputs
    modeltype = args.modeltype
    outfile = args.outfile
    zerodats = args.zerodats
    testval = args.testval

    if outfile is None:
        outfile = modelfile1.split('/')[-1]

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
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))
    drop()

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

    prep('loading config... ')
    (modeltype1, mid1, timestart1) = modelfile1.split('_')
    (timestart1, ext1) = timestart1.split('.')
    modeltype1 = modeltype1.split('/')[-1]
    config1 = pickle.load(open(outdir+'/histories/'+modeltype1+'_conf_'+timestart1+'.pkl', 'rb'))
    num_inputs1 = config1['num_input']

    (modeltype2, mid2, timestart2) = modelfile2.split('_')
    (timestart2, ext2) = timestart2.split('.')
    modeltype2 = modeltype2.split('/')[-1]
    config2 = pickle.load(open(outdir+'/histories/'+modeltype2+'_conf_'+timestart2+'.pkl', 'rb'))
    num_inputs2 = config1['num_input']
    drop()

    prep('loading model... ')
    model1 = keras.models.load_model(modelfile1, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer, "SeqSelfAttention":SeqSelfAttention})
    model2 = keras.models.load_model(modelfile2, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer, "SeqSelfAttention":SeqSelfAttention})
    #print(model2.summary())
    drop()

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outfn = outdir+"/predictions/predict-dual-{}.txt".format(outfile.split('.')[0])
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
 
    prep("computing predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        st = timer()
        
        for fid in fid_set:
            seqdata['c'+testval][fid] = comstart #np.asarray([stk])
            
        bg = batch_gen(seqdata, testval, config1, training=False)
        batch = bg.make_batch(fid_set)

        if config1['batch_maker'] == 'datsonly':
            batch_results = gendescr_2inp(model1, batch, comstok, comlen, batchsize, config1, strat='greedy')
        elif config1['batch_maker'] == 'ast':
            batch_results = gendescr_3inp(model1, batch, comstok, comlen, batchsize, config1, strat='greedy')
        elif config1['batch_maker'] == 'ast_threed':
            batch_results = gendescr_4inp(model1, batch, comstok, comlen, batchsize, config1, strat='greedy')
        elif config1['batch_maker'] == 'threed':
            batch_results = gendescr_threed(model1, model2, batch, comstok, comlen, batchsize, config1, config2, strat='greedy')
        elif config1['batch_maker'] == 'graphast':
            batch_results = gendescr_graphast(model1, batch, comstok, comlen, batchsize, config1, strat='greedy')
        elif config1['batch_maker'] == 'graphast_threed':
            batch_results = gendescr_5inp(model1, batch, comstok, comlen, batchsize, config1, strat='greedy')
        else:
            print('error: invalid batch maker')
            sys.exit()

        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    outf.close()        
    drop()
