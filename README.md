# Attn-to-FC

This repository contains the code for Attn-to-FC.

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
LeClair, A., McMillan, C., "Recommendations for Datasets for Source Code Summarization", in Proc. of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL'19), Short Research Paper Track, Minneapolis, USA, June 2-7, 2019.

(The raw data is available here: http://leclair.tech/data/funcom/)  

Extract the dataset to a directory (/nfs/projects/ is the assumed default) so that you have a directory structure:  
/nfs/projects/attn-to-fc/data/standard/dataset.pkl  
etc. in accordance with the files described on the site above.

To be consistent with defaults, create the following directories:  
/nfs/projects/attn-to-fc/data/outdir/models/  
/nfs/projects/attn-to-fc/data/outdir/histories/  
/nfs/projects/attn-to-fc/data/outdir/predictions/  

### Step 2: Train a Model

```console
you@server:~/dev/attn-to-fc$ time python3 train.py --model-type=ast-attendgru-fc --gpu=0
```

Model types are defined in model.py. The 10 digit integer at the end of the file is the epoch time at which training started, and is used to connect model, prediction, configuration and history data.  For example, training ast-attendgru-fc to epoch 8 would produce:

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

A good baseline for initial work is the attendgru model.  It trains relatively quickly: about 45 minutes per epoch using batch size 200 on a single Quadro P5000.

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

Note that CPU prediction is possible in principle, but by default all the models use CuDNNGRU instead of standard GRU, which necessitates using a GPU during prediction.
It is important to note that the data argument needs to be the same between running train.py and test.py.
The ICSE'20 submission versions (prediction files) are all included in the predictions directory in this repository.

### Step 4: Vizualization

```console
you@server:~/dev/attn-to-fc$ time python3 my_get_activations.py /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru-fc_E07_1565109688.h5 /nfs/projects/attn-to-fc/data/outdir/models/ast-attendgru_E04_1554124260.h5 --fid=26052502
```

This will output all the activation map of the attention matrix in ast-attendgru-fc as well as the ast-attendgru matrix. This is designed to allow comparison between models that only differ with the addition of file context information.
It takes 2 models files as input, the first is the model with file context information while the second is the model without file context information. It also takes the function id (fid) as input.
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
