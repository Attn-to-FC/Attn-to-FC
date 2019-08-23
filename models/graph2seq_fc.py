from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.optimizers import RMSprop, Adamax, Adam
from keras.engine.topology import Layer
import keras
import keras.utils
import keras.backend as K
import tensorflow as tf

from custom.graphlayers import OurCustomGraphLayer

class Graph2SeqFCModel:
    def __init__(self, config):
        
        # data length in dataset is 20+ functions per file, but we can elect to reduce
        # that length here, since myutils reads this length when creating the batches
        config['sdatlen'] = 10
        config['stdatlen'] = 25
        
        config['tdatlen'] = 50

        config['maxastnodes'] = 100
        config['asthops'] = 1
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['maxastnodes']

        self.config['batch_maker'] = 'graphast_threed'
        self.config['num_input'] = 5
        self.config['num_output'] = 1
        self.config['use_tdats'] = True

        self.embdims = 100
        self.smldims = 100
        self.recdims = 100
        self.tdddims = 200

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        sdat_input = Input(shape=(self.sdatlen, self.config['stdatlen']))
        com_input = Input(shape=(self.comlen,))
        smlnode_input = Input(shape=(self.smllen,))
        smledge_input = Input(shape=(self.smllen, self.smllen))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)
        
        # note on embedding spaces below
        # in this model, the embedding is shared between tdats, sdats, and ast nodes
        # OurModel8 model is identical to this model, but separates the embedding space for ast nodes
        # the results were that OurModel8 performed much worse than OurModel7 and took far longer to train
        
        #se = Embedding(output_dim=self.smldims, input_dim=self.smlvocabsize, mask_zero=False)(smlnode_input)
        se = tdel(smlnode_input)

        tenc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)
        
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=tstate_h)

        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)
        tcontext = dot([tattn, tencout], axes=[2, 1])

        semb = TimeDistributed(tdel)
        sde = semb(sdat_input)
        
        senc = TimeDistributed(CuDNNGRU(int(self.recdims)))
        senc = senc(sde)

        sattn = dot([decout, senc], axes=[2, 2])
        sattn = Activation('softmax')(sattn)
        scontext = dot([sattn, senc], axes=[2, 1])

        wrknodes = se
        for k in range(self.config['asthops']):
            astwork = OurCustomGraphLayer()([wrknodes, smledge_input])
            astwork = concatenate([astwork, wrknodes]) # combine the new node vectors with the previous iteration
            astwork = Dense(self.embdims)(astwork) # use a dense layer to squash back to proper dimension
            wrknodes = astwork
        
        #astwork = CuDNNGRU(self.recdims, return_sequences=True)(astwork, initial_state=tstate_h)
        
        # attend decoder words to nodes in ast
        aattn = dot([decout, astwork], axes=[2, 2])
        aattn = Activation('softmax')(aattn)
        acontext = dot([aattn, astwork], axes=[2, 1])

        context = concatenate([scontext, tcontext, decout, acontext])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, sdat_input, com_input, smlnode_input, smledge_input], outputs=out1)
        
        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, clipnorm=20.), metrics=['accuracy'])
        return self.config, model
