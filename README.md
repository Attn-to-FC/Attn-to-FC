# Attn-to-FC

This repository contains the public release code for Attn-to-FC, a tool for source code summarization using file context information.

### Publication related to this work:
Haque, S., LeClair, A., Wu, L., McMillan, C., "Improved Automatic Summarization of Subroutines via Attention to File Context", in Proc. of the 17th International Conference on Mining Software Repositories (MSR ’20), October 5–6, 2020, Seoul, Republic of Korea. 

### Example Output
Example output from the ast-attendgru-fc model compared to ast-attendgru model and reference summaries written by humans:

PROTOTYPE OUTPUT WITH FILE CONTEXT - PROTOTYPE OUTPUT WITHOUT FILE CONTEXT - HUMAN REFERENCE

sets the intermediate value for this flight - sets the intermediate value for this <UNK> - sets the intermediate value for this flight
returns a string representation of this exception - returns a string representation of this object - returns a string representation of this exception

## USAGE

### Step 0: Dependencies

We assume Ubuntu 18.04, Python 3.6.7, Keras 2.2.4, TensorFlow 1.14.
Check out the readme file in the Transformer directory on the requirements to run the transformer model.

### Step 1: Obtain Dataset

We use the dataset of 2.1m Java methods and method comments, already cleaned and separated into training/val/test sets by LeClair et al.

(Their raw data was downloaded from: http://leclair.tech/data/funcom/)  

Extract the dataset to a directory (/nfs/projects/ is the assumed default) so that you have a directory structure:  
/nfs/projects/attn-to-fc/data/standard/dataset.pkl.

The data for the code2seq and graph2seq models are organized in a different pickle file and can be obtained from the following link </br>
https://attntofcicse2020.s3.us-east-2.amazonaws.com/data.zip

Extract the dataset in the /nfs/projects/attn-to-fc/data directory.
Notice that this directory also contains an outdir child directory. 
This directory contains the model files, configuration files and prediction outputs of the models listed in table 1 of the paper.

Therefore, the default directory structure should be: </br>
```/nfs/projects/attn-to-fc/data/standard``` which contains the dataset obtained from LeClair et al. </br>
```/nfs/projects/attn-to-fc/data/graphast_data``` which contains the dataset compatible for code2seq and graph2seq </br>
```/nfs/projects/attn-to-fc/data/outdir``` which contains the model files, configuration files, prediction files and have the following structure:</br>
```
/nfs/projects/attn-to-fc/data/outdir/models/  
/nfs/projects/attn-to-fc/data/outdir/histories/  
/nfs/projects/attn-to-fc/data/outdir/predictions/  
/nfs/projects/attn-to-fc/data/outdir/viz/
```

If you choose to have a different directory structure, please make the necessary changes in myutils.py (line 14), predict.py (line 202 and 203), train.py (line 82 and 83), my_get_activations.py (line 37, 65, 66, 101, 102, 103, 144, 145, 146, 184, 185, 227, 228, 229, 245, 246, 280, 281, 338, 363, 364, 365, 440, 446), bleu.py (line 55 and 56), rougemetric.py (line 110 and 111), astpathmaker.py (line 20)



### Step 2: Train a Model

```console
you@server:~/dev/attn-to-fc$ time python3 train.py --model-type=ast-attendgru-fc --gpu=0
```

Model types are defined in model.py. 
The attendgru and ast-attendgru model used was borrowed from the work of LeClair et al. We thank them for making their code open source and their repository accessible to everyone.<br />
Alexander LeClair, Siyuan Jiang, and Collin McMillan. 2019. A neural model for generating natural language summaries of program subroutines. In Proceedings of the 41st International Conference on Software Engineering. IEEE Press, 795–806.<br />
https://arxiv.org/abs/1902.01954

Their github repository link:
https://github.com/mcmillco/funcom

The graph2seq model was our faithful reimplementation of Xu et. al. <br />
Kun Xu, Lingfei Wu, Zhiguo Wang, Yansong Feng, Michael Witbrock, and Vadim Sheinin. 2018.  Graph2seq: Graph to sequence learning with attention-based neural networks. Conference on Empirical Methods in Natural Language Processing (2018).<br />
https://arxiv.org/abs/1804.00823

The code2seq model was our faithful reimplementation of Alon et al.<br />
Uri Alon, Shaked Brody, Omer Levy, and Eran Yahav. 2019. Code2seq: Generating sequences from structured representations of code. International Conference on Learning Representations (2019).<br />
https://arxiv.org/abs/1808.01400

For all these models, we added file context information to implement our own custom models. These models can be found in models/attendgru_fc.py, models/atfilecont.py, models/graph2seq_fc.py and models/code2seq_fc.py

The 10 digit integer at the end of the file is the epoch time at which training started, and is used to connect model, prediction, configuration and history data.  For example, training ast-attendgru-fc (model found in models/atfilecont.py) to epoch 8 would produce:

/nfs/projects/attn-to-fc/data/outdir/histories/ast-attendgru-fc_conf_1565109688.pkl  
/nfs/projects/attn-to-fc/data/outdir/histories/ast-attendgru-fc_hist_1565109688.pkl  
/nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E01_1565109688.h5  
/nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E02_1565109688.h5  
/nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E03_1565109688.h5  
/nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E04_1565109688.h5  
/nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E05_1565109688.h5  
/nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E06_1565109688.h5  
/nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E07_1565109688.h5  
/nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E08_1565109688.h5 


Here are the list of models and their corresponding trained model filenames that were used to obtain the bleu scores as listed in table 1 of the paper:

| Model Name      | Model Path |
| ----------- | ----------- |
| attendgru      | /nfs/projects/attn-to-fc/data/outdir/models/attendgru_E04_1565797619.h5       |
| ast-attendgru   | /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru_E04_1554133793.h5        |
| graph2seq      | /nfs/projects/attn-to-fc/data/outdir/models/graph2seq_E04_1554124260.h5 |
| code2seq      | /nfs/projects/attn-to-fc/data/outdir/models/code2seq_E04_1565726816.h5       |
| attendgru+FC   | /nfs/projects/attn-to-fc/data/outdir/models/attendgru-fc_E05_1564348142.h5        |
| astattendgru+FC      | /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E07_1565109688.h5 |
| graph2seq+FC      | /nfs/projects/attn-to-fc/data/outdir/models/graph2seq-fc_E04_1563279697.h5       |
| code2seq+FC   | /nfs/projects/attn-to-fc/data/outdir/models/code2seq-fc_E04_1565726584.h5        |
| ast-attendgru-ablation      | /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru_E08_1566229103.h5       |
| ast-attendgru+FC-ablation   | /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E06_1566229294.h5        |

```console
you@server:~/dev/attn-to-fc$ time python3 train.py --model-type=ast-attendgru-fc --gpu=0 --help
```
This will output the list of input arguments that can be passed via the command line to train the model.

### Step 3: Inference / Prediction

```console
you@server:~/dev/attn-to-fc$ time python3 predict.py /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E07_1565109688.h5 --gpu=0
```

The only necessary input to predict.py on the command line is the model file.  Output predictions will be written to a file e.g.:

/nfs/projects/attn-to-fc/data/outdir/predictions/predict-ast-attendgru-fc_E07_1565109688.txt

Note that by default all the models use CuDNNGRU instead of standard GRU, so using a GPU is necessary during prediction.<br />
It is important to note that the data argument needs to be the same between running train.py and test.py.<br />
The ICSE'20 submission versions (prediction files) are all included in the predictions directory in this repository.

Note that predictions/predict-ast-attendgru_E08_1566229103.txt and predictions/predict-ast-attendgru-fc_E06_1566229294.txt were the predictions obtained from the ablation study.

### Step 4: Vizualization

```console
you@server:~/dev/attn-to-fc$ time python3 my_get_activations.py /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E07_1565109688.h5 /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru_E04_1554124260.h5 --fid=26052502
```

This will output all the activation map of the attention matrix in ast-attendgru-fc as well as the ast-attendgru matrix. This is designed to allow comparison between models that only differ with the addition of file context information.<br />
It takes 2 models files as input, the first is the model with file context information while the second is the model without file context information. It also takes the function id (fid) as input.<br />
Currently this can be run for only 1 fid at a time.

```console
you@server:~/dev/attn-to-fc$ time python3 my_get_activations.py --help
```

This will output the list of input arguments that can be passed via the command line and how they can be used to generate the desired activation map.

### Step 5: Calculate Metrics

```console
you@server:~/dev/attn-to-fc$ time python3 bleu.py /nfs/projects/attn-to-fc/data/outdir/predictions/predict-ast-attendgru-fc_E07_1565109688.txt
```

This will output a BLEU score for the prediction file.

Similarly,
```console
you@server:~/dev/attn-to-fc$ time python3 rougemetric.py /nfs/projects/attn-to-fc/data/outdir/predictions/predict-ast-attendgru-fc_E07_1565109688.txt
```

This will output a rouge score for the prediction file.
