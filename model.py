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
    	# base attention GRU model based on LeCLair et al.
        mdl = AttentionGRUModel(config)
    elif modeltype == 'ast-attendgru':
    	# attention GRU model with added AST information based on LeClair et al. 
        mdl = xtra(config)
    elif modeltype == 'attendgru-fc':
        # our attention GRU model with added file context
        mdl = attendgrufc(config)
    elif modeltype == 'graph2seq':
        # our implementation of base graph2seq model based on Xu et al.
        mdl = Graph2SeqModel(config)
    elif modeltype == 'graph2seq-fc':
        # our graph2seq model with added file context information
        mdl = graph2seqfc(config)
    elif modeltype == 'ast-attendgru-fc':
        # our attention GRU model with both AST and file context information added
        mdl = xtrafc(config)
    elif modeltype == 'code2seq-fc':
        # our code2seq model with custom file context information added
        mdl = code2seqfc(config)
    elif modeltype == 'code2seq':
        # our implementation of base code2seq model based on Alon et al.
        mdl = Code2SeqModel(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
