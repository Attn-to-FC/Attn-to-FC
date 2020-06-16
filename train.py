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

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

import keras
import keras.utils
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
import keras.backend as K
from model import create_model
from myutils import prep, drop, batch_gen, init_tf, seq2sent
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


class HistoryCallback(Callback):
    
    def setCatchExit(self, outdir, modeltype, timestart, mdlconfig):
        self.outdir = outdir
        self.modeltype = modeltype
        self.history = {}
        self.timestart = timestart
        self.mdlconfig = mdlconfig
        
        atexit.register(self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
        signal.signal(signal.SIGINT, self.handle_exit)
    
    def handle_exit(self, *args):
        if len(self.history.keys()) > 0:
            try:
                fn = outdir+'/histories/'+self.modeltype+'_hist_'+str(self.timestart)+'.pkl'
                histoutfd = open(fn, 'wb')
                pickle.dump(self.history, histoutfd)
                print('saved history to: ' + fn)
                
                fn = outdir+'/histories/'+self.modeltype+'_conf_'+str(self.timestart)+'.pkl'
                confoutfd = open(fn, 'wb')
                pickle.dump(self.mdlconfig, confoutfd)
                print('saved config to: ' + fn)
            except Exception as ex:
                print(ex)
                traceback.print_exc(file=sys.stdout)
        sys.exit()
    
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


if __name__ == '__main__':

    timestart = int(round(time.time()))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--model-type', dest='modeltype', type=str, default='vanilla')
    parser.add_argument('--with-multigpu', dest='multigpu', action='store_true', default=False)
    parser.add_argument('--zero-dats', dest='zerodats', type=str, default='no')
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/attn-to-fc/data/standard')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/attn-to-fc/data/outdir') 
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')
    parser.add_argument('--datfile', dest='datfile', type=str, default='dataset.pkl')
    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    multigpu = args.multigpu
    zerodats = args.zerodats
    datfile = args.datfile

    print(zerodats)
    if zerodats == 'yes':
        zerodats = True
    else:
        zerodats = False
    print(zerodats)

    K.set_floatx(args.dtype)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_loglevel
    
    sys.path.append(dataprep)
    import tokenizer

    init_tf(gpu)

    prep('loading tokenizers... ')
    tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    sdatstok = pickle.load(open('%s/sdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
    drop()

    prep('loading sequences... ')
    #seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))
    seqdata = pickle.load(open('%s/%s' % (dataprep, datfile), 'rb'))
    drop()

    if zerodats:
        v = np.zeros(100)
        for key, val in seqdata['dttrain'].items():
            seqdata['dttrain'][key] = v

        for key, val in seqdata['dtval'].items():
            seqdata['dtval'][key] = v
    
        for key, val in seqdata['dttest'].items():
            seqdata['dttest'][key] = v


    steps = int(len(seqdata['ctrain'])/batch_size)+1
    #steps = 1
    valsteps = int(len(seqdata['cval'])/100)+1
    #valsteps = 1
    
    tdatvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size

    print('tdatvocabsize %s' % (tdatvocabsize))
    print('comvocabsize %s' % (comvocabsize))
    print('smlvocabsize %s' % (smlvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps*batch_size))
    print('vaidation data size {}'.format(valsteps*100))
    print('------------------------------------------')

    config = dict()
    
    config['tdatvocabsize'] = tdatvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    try:
        config['comlen'] = len(list(seqdata['ctrain'].values())[0])
        config['tdatlen'] = len(list(seqdata['dttrain'].values())[0])
        config['sdatlen'] = seqdata['config']['sdatlen']
        config['smllen'] = len(list(seqdata['strain'].values())[0])
    except KeyError:
        pass # some configurations do not have all data, which is fine
    
    config['multigpu'] = multigpu
    config['batch_size'] = batch_size

    prep('creating model... ')
    config, model = create_model(modeltype, config)
    drop()

    print(model.summary())
    

    gen = batch_gen(seqdata, 'train', config)
    #checkpoint = ModelCheckpoint(outdir+'/'+modeltype+'_E{epoch:02d}_TA{acc:.2f}_VA{val_acc:.2f}_VB{val_bleu:}.h5', monitor='val_loss')
    checkpoint = ModelCheckpoint(outdir+'/models/'+modeltype+'_E{epoch:02d}_'+str(timestart)+'.h5')
    savehist = HistoryCallback()
    savehist.setCatchExit(outdir, modeltype, timestart, config)
    
    valgen = batch_gen(seqdata, 'val', config)

    # If you want it to calculate BLEU Score after each epoch use callback_valgen and test_cb
    #####
    #callback_valgen = batch_gen_train_bleu(seqdata, comvocabsize, 'val', modeltype, batch_size=batch_size)
    #test_cb = mycallback(callback_valgen, steps)
    #####
    callbacks = [ checkpoint, savehist ]

    try:
        history = model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=8, workers=1, use_multiprocessing=False, callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
    except Exception as ex:
        print(ex)
        traceback.print_exc(file=sys.stdout)
