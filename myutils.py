import sys
import javalang
from timeit import default_timer as timer
import keras
import numpy as np
import tensorflow as tf
import networkx as nx
import random

# do NOT import keras in this header area, it will break predict.py
# instead, import keras as needed in each function

# TODO refactor this so it imports in the necessary functions
dataprep = '/nfs/projects/attn-to-fc/data/standard'
sys.path.append(dataprep)
import tokenizer

start = 0
end = 0

def init_tf(gpu, horovod=False):
    from keras.backend.tensorflow_backend import set_session
    
    config = tf.ConfigProto()
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu

    set_session(tf.Session(config=config))

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()

def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))
            
class batch_gen(keras.utils.Sequence):
    def __init__(self, seqdata, tt, config, training=True):
        self.comvocabsize = config['comvocabsize']
        self.tt = tt
        self.batch_size = config['batch_size']
        self.seqdata = seqdata
        self.allfids = list(seqdata['dt%s' % (tt)].keys())
        self.num_inputs = config['num_input']
        self.config = config
        self.training = training
        
        random.shuffle(self.allfids) # actually, might need to sort allfids to ensure same order

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchfids = self.allfids[start:end]
        return self.make_batch(batchfids)

    def make_batch(self, batchfids):
        if self.config['batch_maker'] == 'datsonly':
            return self.divideseqs(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'ast':
            return self.divideseqs_ast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'ast_threed':
            return self.divideseqs_ast_threed(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'threed':
            return self.divideseqs_threed(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'graphast':
            return self.divideseqs_graphast(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'graphast_threed':
            return self.divideseqs_graphast_threed(batchfids, self.seqdata, self.comvocabsize, self.tt)
        elif self.config['batch_maker'] == 'pathast_threed':
            return self.divideseqs_pathast_threed(batchfids, self.seqdata, self.comvocabsize, self.tt)
        else:
            return None

    def __len__(self):
        #if self.num_inputs == 4:
        return int(np.ceil(len(list(self.seqdata['dt%s' % (self.tt)]))/self.batch_size))
        #else:
        #    return int(np.ceil(len(list(self.seqdata['d%s' % (self.tt)]))/self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.allfids)

    def divideseqs(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        datseqs = list()
        comseqs = list()
        comouts = list()
        
        fiddat = dict()

        for fid in batchfids:
            wdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            
            wdatseq = wdatseq[:self.config['tdatlen']]
            
            
            if not self.training:
                fiddat[fid] = [wdatseq, wcomseq]
            else:
                for i in range(len(wcomseq)):
                    datseqs.append(wdatseq)
                    comseq = wcomseq[:i]
                    comout = keras.utils.to_categorical(wcomseq[i], num_classes=comvocabsize)
                    
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(np.asarray(comseq))
                    comouts.append(np.asarray(comout))

        datseqs = np.asarray(datseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return [[datseqs, comseqs], comouts]

    def divideseqs_ast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        datseqs = list()
        comseqs = list()
        smlseqs = list()
        comouts = list()
        
        fiddat = dict()

        for fid in batchfids:

            wdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]

            wdatseq = wdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wdatseq, wcomseq, wsmlseq]
            else:
                for i in range(0, len(wcomseq)):
                    datseqs.append(wdatseq)
                    smlseqs.append(wsmlseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        datseqs = np.asarray(datseqs)
        smlseqs = np.asarray(smlseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            return [[datseqs, comseqs, smlseqs], comouts]

    def divideseqs_ast_threed(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        smlseqs = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlseq = seqdata['s%s' % (tt)][fid]

            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # the dataset contains 20+ functions per file, but we may elect
            # to reduce that amount for a given model based on the config
            newlen = self.config['sdatlen']-len(wsdatseq)
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'],:]

            wsmlseq = wsmlseq[:self.config['smllen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wcomseq, wsmlseq]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    smlseqs.append(wsmlseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        smlseqs = np.asarray(smlseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, comseqs, smlseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, sdatseqs, comseqs, smlseqs], comouts]

    def divideseqs_threed(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        comouts = list()
        
        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]

            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # the dataset contains 20+ functions per file, but we may elect
            # to reduce that amount for a given model based on the config
            newlen = self.config['sdatlen']-len(wsdatseq)
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'],:]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wcomseq]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, comseqs], [comouts, comouts]]
            else:
                return [[tdatseqs, sdatseqs, comseqs], comouts]

    def divideseqs_graphast(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        comseqs = list()
        smlnodes = list()
        smledges = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                continue

            # crop/expand ast sequence
            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)
            
            # crop/expand ast adjacency matrix to dense
            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp)

            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            if not self.training:
                fiddat[fid] = [wtdatseq, wcomseq, wsmlnodes, wsmledges]
            else:
                for i in range(0, len(wcomseq)):
                    if(self.config['use_tdats']):
                        tdatseqs.append(wtdatseq)
                    smlnodes.append(wsmlnodes)
                    smledges.append(wsmledges)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        if(self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        smlnodes = np.asarray(smlnodes)
        smledges = np.asarray(smledges)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, comseqs, smlnodes, smledges], [comouts, comouts]]
            else:
                if(self.config['use_tdats']):
                    return [[tdatseqs, comseqs, smlnodes, smledges], comouts]
                else:
                    return [[comseqs, smlnodes, smledges], comouts]


    def divideseqs_graphast_threed(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        smlnodes = list()
        smledges = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0] > 1000):
                continue

            # crop/expand ast sequence
            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)
            
            # crop/expand ast adjacency matrix to dense
            wsmledges = np.asarray(wsmledges.todense())
            wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            wsmledges = np.int32(tmp)

            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # the dataset contains 20+ functions per file, but we may elect
            # to reduce that amount for a given model based on the config
            newlen = self.config['sdatlen']-len(wsdatseq)
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'],:]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wcomseq, wsmlnodes, wsmledges]
            else:
                for i in range(0, len(wcomseq)):
                    if(self.config['use_tdats']):
                        tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    smlnodes.append(wsmlnodes)
                    smledges.append(wsmledges)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        if(self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        smlnodes = np.asarray(smlnodes)
        smledges = np.asarray(smledges)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, comseqs, smlnodes, smledges], [comouts, comouts]]
            else:
                if(self.config['use_tdats']):
                    return [[tdatseqs, sdatseqs, comseqs, smlnodes, smledges], comouts]
                else:
                    return [[sdatseqs, comseqs, smlnodes, smledges], comouts]

    def idx2tok(self, nodelist, path):
        out = list()
        for idx in path:
            out.append(nodelist[idx])
        return out

    def divideseqs_pathast_threed(self, batchfids, seqdata, comvocabsize, tt):
        import keras.utils
        
        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        smlpaths = list()
        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            wsmledges = seqdata['s%s_edges' % (tt)][fid]
            if (wsmledges.shape[0]):
                continue

            # crop/expand ast sequence
            #wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            #tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            #tmp[:wsmlnodes.shape[0]] = wsmlnodes
            #wsmlnodes = np.int32(tmp)
            
            # crop/expand ast adjacency matrix to dense
            #wsmledges = np.asarray(wsmledges.todense())
            #wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
            #tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            #tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
            #wsmledges = np.int32(tmp)

            g = nx.from_numpy_matrix(wsmledges.todense())
            astpaths = nx.all_pairs_shortest_path(g, cutoff=self.config['pathlen'])
            wsmlpaths = list()

            for astpath in astpaths:
                source = astpath[0]
                
                if len([n for n in g.neighbors(source)]) > 1:
                    continue
                
                for path in astpath[1].values():
                    if len([n for n in g.neighbors(path[-1])]) > 1:
                        continue # ensure only terminals as in Alon et al
                    
                    if len(path) > 1 and len(path) <= self.config['pathlen']:
                        newpath = self.idx2tok(wsmlnodes, path)
                        tmp = [0] * (self.config['pathlen'] - len(newpath))
                        newpath.extend(tmp)
                        wsmlpaths.append(newpath)
            
            random.shuffle(wsmlpaths) # Alon et al stipulate random selection of paths
            wsmlpaths = wsmlpaths[:self.config['maxpaths']] # Alon et al use 200, crop/expand to size
            if len(wsmlpaths) < self.config['maxpaths']:
                wsmlpaths.extend([[0]*self.config['pathlen']] * (self.config['maxpaths'] - len(wsmlpaths)))
            wsmlpaths = np.asarray(wsmlpaths)

            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            # the dataset contains 20+ functions per file, but we may elect
            # to reduce that amount for a given model based on the config
            newlen = self.config['sdatlen']-len(wsdatseq)
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'],:]

            if not self.training:
                fiddat[fid] = [wtdatseq, wsdatseq, wcomseq, wsmlpaths]
            else:
                for i in range(0, len(wcomseq)):
                    if(self.config['use_tdats']):
                        tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    smlpaths.append(wsmlpaths)
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        if(self.config['use_tdats']):
            tdatseqs = np.asarray(tdatseqs)
        if(self.config['use_sdats']):
            sdatseqs = np.asarray(sdatseqs)
        smlpaths = np.asarray(smlpaths)
        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if not self.training:
            return fiddat
        else:
            if self.config['num_output'] == 2:
                return [[tdatseqs, sdatseqs, comseqs, smlpaths], [comouts, comouts]]
            else:
                if(self.config['use_tdats'] and self.config['use_sdats']):
                    return [[tdatseqs, sdatseqs, comseqs, smlpaths], comouts]
                elif(self.config['use_tdats'] and not self.config['use_sdats']):
                    return [[tdatseqs, comseqs, smlpaths], comouts]
                elif(not self.config['use_tdats'] and self.config['use_sdats']):
                    return [[sdatseqs, comseqs, smlpaths], comouts]
                elif(not self.config['use_tdats'] and not self.config['use_sdats']):
                    return [[comseqs, smlpaths], comouts]
