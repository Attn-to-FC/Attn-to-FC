#Transformer implementation modified for this project.

## USAGE

### Step 0: Dependencies
Ubuntu 18.04, python 3.6.7, Tensor2Tensor, Tensorflow 1.14.0, nltk, javalang, keras2.2.4

### Step 1: Obtain Dataset
Before running the code, create 3 directories here: data_dir, output_dir and tmp_dir. Then, copy the contents of the /nfs/projects/attn-to-fc/data/standard/output to the tmp_dir directory.

### Step 2: Train + Predict + Calculate Bleu score
To run the code, just type ./runmodel.sh on the command line.
