from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.backend import tile, repeat, repeat_elements, squeeze, transpose
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

from models.attendgru import AttentionGRUModel
from models.ast_attendgru_xtra import AstAttentionGRUModel as xtra
from models.attendgru_fc import AttentionGRUFCModel as attendgrufc
from models.graph2seq import Graph2SeqModel
from models.graph2seq_fc import Graph2SeqFCModel as graph2seqfc
from models.atfilecont import AstAttentiongruFCModel as xtrafc
from models.code2seq_fc import Code2SeqFCModel as code2seqfc
from models.code2seq import Code2SeqModel

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attendgru':
    	# base attention GRU model based on Nematus architecture
        mdl = AttentionGRUModel(config)
    elif modeltype == 'ast-attendgru':
    	# attention GRU model with added AST information from srcml. 
        mdl = xtra(config)
    elif modeltype == 'attendgru-fc':
        # sandbox model to try things
        mdl = attendgrufc(config)
    elif modeltype == 'graph2seq':
        # sandbox model to try things
        mdl = Graph2SeqModel(config)
    elif modeltype == 'graph2seq-fc':
        # sandbox model to try things
        mdl = graph2seqfc(config)
    elif modeltype == 'ast-attendgru-fc':
        # sandbox model to try things
        mdl = xtrafc(config)
    elif modeltype == 'code2seq-fc':
        # sandbox model to try things
        mdl = code2seqfc(config)
    elif modeltype == 'code2seq':
        # sandbox model to try things
        mdl = Code2SeqModel(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
