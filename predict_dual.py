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

from models.attendgru import top2, top3, top5

def top2(y1, y2):
    return metrics.top_k_categorical_accuracy(y1, y2, k=2)

def top3(y1, y2):
    return metrics.top_k_categorical_accuracy(y1, y2, k=3)

def top5(y1, y2):
    return metrics.top_k_categorical_accuracy(y1, y2, k=5)

def gendescr_dual(txtmodel, astmodel, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...

    dats, coms, smls = list(zip(*data.values()))
    dats = np.array(dats)
    coms = np.array(coms)
    smls = np.array(smls)

    #dats = np.zeros_like(dats)

    for i in range(1, comlen):
        txtresults = txtmodel.predict([dats, coms], batch_size=batchsize)
        astresults = astmodel.predict([dats, coms, smls], batch_size=batchsize)

        for c, (t, a) in enumerate(zip(txtresults, astresults)):
            #tm = t[np.argmax(t)]
            #am = a[np.argmax(a)]
            #m = np.argmax(t)
            #if(am > tm):
            #    m = np.argmax(a)
            m = np.argmax(np.mean([t, a], axis=0))
            coms[c][i] = m

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_3inp(model, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...
    
    dats, coms, smls = list(zip(*data.values()))
    dats = np.array(dats)
    coms = np.array(coms)
    smls = np.array(smls)

    dats = np.zeros_like(dats)

    for i in range(1, comlen):
        results = model.predict([dats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_2inp(model, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...
    
    dats, coms = list(zip(*data.values()))
    dats = np.array(dats)
    coms = np.array(coms)

    for i in range(1, comlen):
        results = model.predict([dats, coms], batch_size=batchsize)
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
    parser.add_argument('txtmodelfile', type=str, default=None)
    parser.add_argument('astmodelfile', type=str, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='../data/makedataset')
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--num-inputs', dest='numinputs', type=int, default=3)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    txtmodelfile = args.txtmodelfile
    astmodelfile = args.astmodelfile
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    num_inputs = args.numinputs
    modeltype = args.modeltype
    outfile = args.outfile
    challenge = args.challenge
    obf = args.obfuscate
    sbt = args.sbt
    # if outfile is None:
    #     outfile = modeltype

    if outfile is None:
        outfile = txtmodelfile.split('/')[-1]

    if challenge:
        dataprep = '../data/challengeset'

    if obf:
        dataprep = '../data/obfuscation'

    if sbt:
        dataprep = '../data/sbt'

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sys.path.append(dataprep)
    import tokenizer

    prep('loading tokenizers... ')
    datstok = pickle.load(open('%s/dats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    if not sbt:
        smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()
        

    #datstok.set_vocab_size(1000)
    #comstok.set_vocab_size(300)

    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))
    drop()

    allfids = list(seqdata['ctest'].keys())
    datvocabsize = datstok.vocab_size
    comvocabsize = comstok.vocab_size
    if not sbt:
        smlvocabsize = smltok.vocab_size

    datlen = len(seqdata['dtest'][list(seqdata['dtest'].keys())[0]])
    comlen = len(seqdata['ctest'][list(seqdata['ctest'].keys())[0]])
    if not sbt:  
        smllen = len(seqdata['stest'][list(seqdata['stest'].keys())[0]])

    prep('loading model...')
    #try:
    astmodel = keras.models.load_model(astmodelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
    txtmodel = keras.models.load_model(txtmodelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
    #except:
        #print("NO")
        #exit()
        #model = load_model_from_weights(modelfile, modeltype, datvocabsize, comvocabsize, smlvocabsize, datlen, comlen, smllen)
    drop()

    comstart = np.zeros(comlen)
    st = comstok.w2i['<s>']
    comstart[0] = st
    outf = open("./outdir/predict-{}.txt".format(outfile), 'w')
    print("writing to file - outdir/predict-{}.txt".format(outfile))
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
 
    prep("Computing Predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        batch = {}
        st = timer()
        for fid in fid_set:
            dat = seqdata['dtest'][fid]
            if not sbt:
                sml = seqdata['stest'][fid]
             # should be fixed size anyway

            if num_inputs == 3:
                batch[fid] = np.asarray([dat, comstart, sml])
            else:
                batch[fid] = np.asarray([dat, comstart])

        if num_inputs == 3:
            batch_results = gendescr_dual(txtmodel, astmodel, batch, comstok, comlen, batchsize, strat='greedy')
        else:
            batch_results = gendescr_2inp(model, batch, comstok, comlen, batchsize, strat='greedy')

        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))

    outf.close()        
    drop()










    """
    r = random.randint(0, numprocs)
    if modeltype == 'ast-attendgru' or modeltype == 'pretrained':
        wrkunits[r].append((fid, dat, comlen, sml))
    else:
        wrkunits[r].append((fid, dat, comlen))

    if(c > 0 and c % 1000 == 0):
        
        for rets in pool.map(gendescrs, wrkunits.values()):
            #for rets in allrets:
                for ret in rets:
                    (fid, pred) = ret
                    predsfile.write('%s\t%s\n' % (fid, pred))
                    predsfile.flush()
        
        wrkunits = collections.defaultdict(list)
        
        et = timer()
        statusout('%s/%ss, ' % (c, round(et-st, 1)))
        st = timer()
        
    predsfile.close()
    drop()

    #batch_size = 1800
    #steps = int(len(seqdata['coms_test_seqs'])/batch_size)+1

    #gen = createbatchgen(seqdata, comvocabsize, 'test', batch_size=batch_size)
    #try:
    #    score = model.evaluate_generator(gen, steps=steps, verbose=1, max_queue_size=2)
    #    print('loss: %s, accuracy: %s' % (score[0], score[1]))
    #except Exception as ex:
    #    print(ex)
    #    traceback.print_exc(file=sys.stdout)


    """
