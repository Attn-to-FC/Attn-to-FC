from keras import models
import numpy as np
import keras.backend as K
import pickle
import random
import argparse
import os
import sys

from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word, init_tf

from timeit import default_timer as timer

import tensorflow as tf
import keras

from custom.graphlayers import OurCustomGraphLayer

# get_activations and display activations based on functions from
# https://github.com/philipperemy/keras-visualize-activations

def gendescr_2inp(model, data, comstok, comlen, batchsize, config, strat, beamwidth, outfile, stopword):
    # right now, only greedy search is supported...
    fid = [*data.keys()]

    tdats, coms = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)

    for i in range(1, stopword):
        results = model.predict([tdats, coms], batch_size=batchsize)

        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    act1 = get_activations(model, [tdats, coms], layer_name='activation_1')
    act1_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act1-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act1_prob_file = open(act1_softmax_val_path, 'w')
    for j in act1:
        display_activations(j, 'tdats_activation', fid, act1_prob_file, outfile, stopword)
    act1_prob_file.close()

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_3inp(model, data, comstok, comlen, batchsize, config, strat, beamwidth, outfile, stopword):
    # right now, only greedy search is supported...
    fid = [*data.keys()]
    
    tdats, coms, smls = list(zip(*data.values()))
    tdats = np.array(tdats)
    coms = np.array(coms)
    smls = np.array(smls)

    for i in range(1, stopword):
        results = model.predict([tdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    act1 = get_activations(model, [tdats, coms, smls], layer_name='activation_1')
    act2 = get_activations(model, [tdats, coms, smls], layer_name='activation_2')
    act1_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act1-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act2_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act2-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act1_prob_file = open(act1_softmax_val_path, 'w')
    act2_prob_file = open(act2_softmax_val_path, 'w')
    for j in act1:
        display_activations(j, 'tdats_activation', fid, act1_prob_file, outfile, stopword)
    for j in act2:
        display_activations(j, 'ast_activation', fid, act2_prob_file, outfile, stopword)
    act1_prob_file.close()
    act2_prob_file.close()

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_4inp(model, data, comstok, comlen, batchsize, config, strat, beamwidth, outfile, stopword):
    # right now, only greedy search is supported...
    fid = [*data.keys()]

    tdats, sdats, coms, smls = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    smls = np.array(smls)

    for i in range(1, stopword):
        results = model.predict([tdats, sdats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    act1 = get_activations(model, [tdats, sdats, coms, smls], layer_name='activation_1')
    act2 = get_activations(model, [tdats, sdats, coms, smls], layer_name='activation_2')
    act3 = get_activations(model, [tdats, sdats, coms, smls], layer_name='activation_3')
    act1_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act1-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act2_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act2-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act3_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act3-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act1_prob_file = open(act1_softmax_val_path, 'w')
    act2_prob_file = open(act2_softmax_val_path, 'w')
    act3_prob_file = open(act3_softmax_val_path, 'w')
    for j in act1:
        display_activations(j, 'ast_activation', fid, act1_prob_file, outfile, stopword)
    for j in act2:
        display_activations(j, 'tdats_activation', fid, act2_prob_file, outfile, stopword)
    for j in act3:
        display_activations(j, 'sattn_activation', fid, act3_prob_file, outfile, stopword)
    act1_prob_file.close()
    act2_prob_file.close()
    act3_prob_file.close()

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_5inp(model, data, comstok, comlen, batchsize, config, strat, beamwidth, outfile, stopword):
    # right now, only greedy search is supported...
    fid = [*data.keys()]

    tdats, sdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)


    for i in range(1, stopword):
        results = model.predict([tdats, sdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    act1 = get_activations(model, [tdats, sdats, coms, wsmlnodes, wsmledges], layer_name='activation_1')
    act2 = get_activations(model, [tdats, sdats, coms, wsmlnodes, wsmledges], layer_name='activation_2')
    act3 = get_activations(model, [tdats, sdats, coms, wsmlnodes, wsmledges], layer_name='activation_3')
    act1_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act1-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act2_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act2-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act3_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act3-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act1_prob_file = open(act1_softmax_val_path, 'w')
    act2_prob_file = open(act2_softmax_val_path, 'w')
    act3_prob_file = open(act3_softmax_val_path, 'w')
    for j in act1:
        display_activations(j, 'tdats_activation', fid, act1_prob_file, outfile, stopword)
    for j in act2:
        display_activations(j, 'sdats_activation', fid, act2_prob_file, outfile, stopword)
    for j in act3:
        display_activations(j, 'ast_activation', fid, act3_prob_file, outfile, stopword)
    act1_prob_file.close()
    act2_prob_file.close()
    act3_prob_file.close()

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_graphast(model, data, comstok, comlen, batchsize, config, strat, beamwidth, outfile, stopword):
    # right now, only greedy search is supported...
    fid = [*data.keys()]

    tdats, coms, wsmlnodes, wsmledges = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wsmledges = np.array(wsmledges)

    for i in range(1, stopword):
        results = model.predict([tdats, coms, wsmlnodes, wsmledges], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    act1 = get_activations(model, [tdats, coms, smls], layer_name='activation_1')
    act2 = get_activations(model, [tdats, coms, smls], layer_name='activation_2')
    act1_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act1-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act2_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act2-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act1_prob_file = open(act1_softmax_val_path, 'w')
    act2_prob_file = open(act2_softmax_val_path, 'w')
    for j in act1:
        display_activations(j, 'tdats_activation', fid, act1_prob_file, outfile, stopword)
    for j in act2:
        display_activations(j, 'ast_activation', fid, act2_prob_file, outfile, stopword)
    act1_prob_file.close()
    act2_prob_file.close()

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_pathast(model, data, comstok, comlen, batchsize, config, strat, beamwidth, outfile, stopword):
    # right now, only greedy search is supported...

    fid = [*data.keys()]

    tdats, sdats, coms, wsmlpaths = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    sdats = np.array(sdats)
    wsmlpaths = np.array(wsmlpaths)

    #print(sdats)

    for i in range(1, stopword):
        if(config['use_sdats']):
            results = model.predict([tdats, sdats, coms, wsmlpaths], batch_size=batchsize)
        else:
            results = model.predict([tdats, coms, wsmlpaths], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    if (config['use_sdats']):
        act1 = get_activations(model, [tdats, sdats, coms, wsmlpaths], layer_name='activation_1')
        act2 = get_activations(model, [tdats, sdats, coms, wsmlpaths], layer_name='activation_2')
        act3 = get_activations(model, [tdats, sdats, coms, wsmlpaths], layer_name='activation_3')
        act1_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act1-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
        act2_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act2-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
        act3_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act3-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
        act1_prob_file = open(act1_softmax_val_path, 'w')
        act2_prob_file = open(act2_softmax_val_path, 'w')
        act3_prob_file = open(act3_softmax_val_path, 'w')
        for j in act1:
            display_activations(j, 'tdats_activation', fid, act1_prob_file, outfile, stopword)
        for j in act2:
            display_activations(j, 'sdats_activation', fid, act2_prob_file, outfile, stopword)
        for j in act3:
            display_activations(j, 'ast_activation', fid, act3_prob_file, outfile, stopword)
        act1_prob_file.close()
        act2_prob_file.close()
        act3_prob_file.close()
    else:
        act1 = get_activations(model, [tdats, coms, wsmlpaths], layer_name='activation_1')
        act2 = get_activations(model, [tdats, coms, wsmlpaths], layer_name='activation_2')
        act1_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act1-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
        act2_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act2-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
        act1_prob_file = open(act1_softmax_val_path, 'w')
        act2_prob_file = open(act2_softmax_val_path, 'w')
        for j in act1:
            display_activations(j, 'tdats_activation', fid, act1_prob_file, outfile, stopword)
        for j in act2:
            display_activations(j, 'ast_activation', fid, act2_prob_file, outfile, stopword)
        act1_prob_file.close()
        act2_prob_file.close()

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def gendescr_threed(model, data, comstok, comlen, batchsize, config, strat, beamwidth, outfile, stopword):
    # right now, only greedy search is supported...
    fid = [*data.keys()]

    tdats, sdats, coms = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)

    for i in range(1, stopword):
        results = model.predict([tdats, sdats, coms], batch_size=batchsize)

        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)
        
    act1 = get_activations(model, [tdats, sdats, coms], layer_name='activation_1')
    act2 = get_activations(model, [tdats, sdats, coms], layer_name='activation_2')
    act1_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act1-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act2_softmax_val_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-act2-stopword-{}.txt'.format(fid[0], outfile.split('.')[0], stopword)
    act1_prob_file = open(act1_softmax_val_path, 'w')
    act2_prob_file = open(act2_softmax_val_path, 'w')
    for j in act1:
        display_activations(j, 'tdats_activation', fid, act1_prob_file, outfile, stopword)
    for j in act2:
        display_activations(j, 'sdats_activation', fid, act2_prob_file, outfile, stopword)
    act1_prob_file.close()
    act2_prob_file.close()

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def get_activations(model, model_inputs, print_shape_only=True, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]


    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps, title, fid_list, act_prob_file, outfile, stopword):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (80, 12)

    for i, activation_map in enumerate(activation_maps):
        fid = fid_list[i]
        img_path = '/nfs/projects/attn-to-fc/data/outdir/viz/{}-{}-{}-stopword-{}.pdf'.format(fid, outfile.split('.')[0], title, stopword)
        act_prob_file.write(str(fid)+'\t'+str(activation_map)+'\n')
        
        activation_map = np.expand_dims(activation_map, axis=0)
        batch_size = activation_map.shape[0]
        assert batch_size == 1, 'One image at a time to visualize.'
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape

        plt.imshow(activation_map[0], interpolation='nearest')
        plt.title(title)
        plt.savefig(img_path)
        cmd = "xpdf {} &".format(img_path)
        os.system(cmd)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modelfilewsdats', type=str, default=None)
    parser.add_argument('modelfilewosdats', type=str, default=None)
    parser.add_argument('--fid', type=int, default=None)
    parser.add_argument('--stopword', type=int, default=None)
    parser.add_argument('--num-procs', dest='numprocs', type=int, default='4')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data1', dest='dataprep', type=str, default='/nfs/projects/attn-to-fc/data/standard')
    parser.add_argument('--data2', dest='dataprep2', type=str, default='/nfs/projects/attn-to-fc/data/standard')
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/attn-to-fc/data/outdir')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=200)
    parser.add_argument('--num-inputs', dest='numinputs', type=int, default=3)
    parser.add_argument('--model-type', dest='modeltype', type=str, default=None)
    parser.add_argument('--strat', dest='strat', type=str, default='greedy')
    parser.add_argument('--beam-width', dest='beamwidth', type=int, default=1)
    parser.add_argument('--zero-dats', dest='zerodats', action='store_true', default=False)
    parser.add_argument('--dtype', dest='dtype', type=str, default='float32')
    parser.add_argument('--tf-loglevel', dest='tf_loglevel', type=str, default='3')

    args = parser.parse_args()
    
    outdir = args.outdir
    dataprep = args.dataprep
    dataprep2 = args.dataprep2
    modelfilewsdats = args.modelfilewsdats
    modelfilewosdats = args.modelfilewosdats
    fid = args.fid
    stopword = args.stopword
    numprocs = args.numprocs
    gpu = args.gpu
    batchsize = args.batchsize
    num_inputs = args.numinputs
    modeltype = args.modeltype
    strat = args.strat
    beamwidth = args.beamwidth
    zerodats = args.zerodats

    outfilewsdats = modelfilewsdats.split('/')[-1]
    outfilewosdats = modelfilewosdats.split('/')[-1]

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

    allfids = list(seqdata['ctest'].keys())
    datvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size

    datlen = len(seqdata['dttest'][list(seqdata['dttest'].keys())[0]])
    comlen = len(seqdata['ctest'][list(seqdata['ctest'].keys())[0]])
    #smllen = len(seqdata['stest'][list(seqdata['stest'].keys())[0]])

    if stopword is None:
        stopword=comlen

    prep('loading config... ')
    (modeltypewsdats, mwsdatsid, timestartwsdats) = modelfilewsdats.split('_')
    (timestartwsdats, extwsdats) = timestartwsdats.split('.')
    modeltypewsdats = modeltypewsdats.split('/')[-1]
    configwsdats = pickle.load(open('/nfs/projects/attn-to-fc/data/outdir/histories/'+modeltypewsdats+'_conf_'+timestartwsdats+'.pkl', 'rb'))
    num_inputswsdats = configwsdats['num_input']

    (modeltypewosdats, mwosdatsid, timestartwosdats) = modelfilewosdats.split('_')
    (timestartwosdats, extwosdats) = timestartwosdats.split('.')
    modeltypewosdats = modeltypewosdats.split('/')[-1]
    configwosdats = pickle.load(open('/nfs/projects/attn-to-fc/data/outdir/histories/'+modeltypewosdats+'_conf_'+timestartwosdats+'.pkl', 'rb'))
    num_inputswosdats = configwosdats['num_input']
    drop()

    prep('loading model... ')
    modelwsdats = keras.models.load_model(modelfilewsdats, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer})
    modelwosdats = keras.models.load_model(modelfilewosdats, custom_objects={"tf":tf, "keras":keras, "OurCustomGraphLayer":OurCustomGraphLayer})
    print(modelwsdats.summary())
    print(modelwosdats.summary())
    drop()

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    batch_sets=[[fid]]

    # cmd = "grep {} /nfs/projects/attn-to-fc/data/outdir/viz/diff_exp_sys2.txt".format(fid)
    # os.system(cmd)
    # print()
    # cmd = "grep {} /nfs/projects/attn-to-fc/data/standard/output/coms.test".format(fid)
    # print('coms for {}'.format(fid))
    # os.system(cmd)
    # print()
    # cmd = "grep {} /nfs/projects/attn-to-fc/data/standard/output/tdats.test".format(fid)
    # print('tdats for {}'.format(fid))
    # os.system(cmd)
    # print()
    # cmd = "grep {} /nfs/projects/attn-to-fc/data/standard/output/sdats.test".format(fid)
    # print('sdats for {}'.format(fid))
    # os.system(cmd)
    # print()
    
    prep("computing predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        st = timer()
        
        for fid in fid_set:
            seqdata['ctest'][fid] = comstart #np.asarray([stk])

        bg = batch_gen(seqdata, 'test', configwsdats, training=False)
        batch = bg.make_batch(fid_set)
        
        if configwsdats['batch_maker'] == 'datsonly':
            batch_resultswsdats = gendescr_2inp(modelwsdats, batch, comstok, comlen, batchsize, configwsdats, strat, beamwidth, outfilewsdats, stopword)
        elif configwsdats['batch_maker'] == 'ast':
            batch_resultswsdats = gendescr_3inp(modelwsdats, batch, comstok, comlen, batchsize, configwsdats, strat, beamwidth, outfilewsdats, stopword)
        elif configwsdats['batch_maker'] == 'ast_threed':
            batch_resultswsdats = gendescr_4inp(modelwsdats, batch, comstok, comlen, batchsize, configwsdats, strat, beamwidth, outfilewsdats, stopword)
        elif configwsdats['batch_maker'] == 'threed':
            batch_resultswsdats = gendescr_threed(modelwsdats, batch, comstok, comlen, batchsize, configwsdats, strat, beamwidth, outfilewsdats, stopword)
        elif configwsdats['batch_maker'] == 'graphast':
            batch_resultswsdats = gendescr_graphast(modelwsdats, batch, comstok, comlen, batchsize, configwsdats, strat, beamwidth, outfilewsdats, stopword)
        elif configwsdats['batch_maker'] == 'graphast_threed':
            batch_resultswsdats = gendescr_5inp(modelwsdats, batch, comstok, comlen, batchsize, configwsdats, strat, beamwidth, outfilewsdats, stopword)
        elif config['batch_maker'] == 'pathast_threed':
            batch_results = gendescr_pathast(modelwsdats, batch, comstok, comlen, batchsize, configwsdats, strat, beamwidth, outfilewsdats, stopword)
        else:
            print('error: invalid batch maker')
            sys.exit()

        for key, val in batch_resultswsdats.items():
            print("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))
    
    prep('loading sequences... ')
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep2), 'rb'))
    drop()

    prep("computing predictions...\n")
    for c, fid_set in enumerate(batch_sets):
        st = timer()
        
        for fid in fid_set:
            seqdata['ctest'][fid] = comstart #np.asarray([stk])

        bg = batch_gen(seqdata, 'test', configwosdats, training=False)
        batch = bg.make_batch(fid_set)

        if configwosdats['batch_maker'] == 'datsonly':
            batch_resultswosdats = gendescr_2inp(modelwosdats, batch, comstok, comlen, batchsize, configwosdats, strat, beamwidth, outfilewosdats, stopword)
        elif configwosdats['batch_maker'] == 'ast':
            batch_resultswosdats = gendescr_3inp(modelwosdats, batch, comstok, comlen, batchsize, configwosdats, strat, beamwidth, outfilewosdats, stopword)
        elif configwosdats['batch_maker'] == 'ast_threed':
            batch_resultswosdats = gendescr_4inp(modelwosdats, batch, comstok, comlen, batchsize, configwosdats, strat, beamwidth, outfilewosdats, stopword)
        elif configwosdats['batch_maker'] == 'threed':
            batch_resultswosdats = gendescr_threed(modelwosdats, batch, comstok, comlen, batchsize, configwosdats, strat, beamwidth, outfilewosdats, stopword)
        elif configwosdats['batch_maker'] == 'graphast':
            batch_resultswosdats = gendescr_graphast(modelwosdats, batch, comstok, comlen, batchsize, configwosdats, strat, beamwidth, outfilewosdats, stopword)
        elif configwosdats['batch_maker'] == 'graphast_threed':
            batch_resultswosdats = gendescr_5inp(modelwosdats, batch, comstok, comlen, batchsize, configwosdats, strat, beamwidth, outfilewosdats, stopword)
        elif config['batch_maker'] == 'pathast_threed':
            batch_results = gendescr_pathast(modelwosdats, batch, comstok, comlen, batchsize, configwosdats, strat, beamwidth, outfilewosdats, stopword)
        else:
            print('error: invalid batch maker')
            sys.exit()

        for key, val in batch_resultswosdats.items():
            print("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, batchsize/(end-st)))
    drop()
