from keras.models import Model
from keras.layers import Input, Maximum, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Bidirectional, CuDNNGRU, RepeatVector, Permute, TimeDistributed, dot
from keras.backend import tile, repeat, repeat_elements
from keras.optimizers import RMSprop, Adamax
import keras
import keras.utils
import tensorflow as tf

class AttentionGRUFCModel:
    def __init__(self, config):
        
        # data length in dataset is 20+ functions per file, but we can elect to reduce
        # that length here, since myutils reads this length when creating the batches
        config['sdatlen'] = 10
        config['stdatlen'] = 25
        
        config['tdatlen'] = 50

        config['smllen'] = 100
        config['3dsmls'] = False
        
        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']

        self.config['batch_maker'] = 'threed'
        self.config['num_input'] = 3
        self.config['num_output'] = 1

        self.embdims = 100
        self.recdims = 256
        self.tdddims = 256

    def create_model(self):
        
        tdat_input = Input(shape=(self.tdatlen,))
        sdat_input = Input(shape=(self.sdatlen, self.config['stdatlen']))
        com_input = Input(shape=(self.comlen,))
        
        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)

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

        context = concatenate([scontext, tcontext, decout])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)

        out = Flatten()(out)
        out1 = Dense(self.comvocabsize, activation="softmax")(out)
        
        model = Model(inputs=[tdat_input, sdat_input, com_input], outputs=out1)
        
        if self.config['multigpu']:
            model = keras.utils.multi_gpu_model(model, gpus=2)

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return self.config, model
