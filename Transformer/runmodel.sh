#!/bin/bash
#
#SBATCH --job-name=runtransformer

time t2t-datagen --t2t_usr_dir=. --data_dir=data_dir --tmp_dir=tmp_dir --problem=code_comment

time t2t-trainer --t2t_usr_dir=. --data_dir=data_dir --model=transformer --problem=code_comment --hparams_set=transformer_base --hparams='batch_size=512' --output_dir=output_dir

time python tdats_remove_fid.py

time t2t-decoder --t2t_usr_dir=. --data_dir=data_dir --problem=code_comment --model=transformer --hparams_set=transformer_base --hparams='batch_size=512' --output_dir=output_dir --decode_hparams='beam_size=1, alpha=0' --decode_from_file=tmp_dir/new_tdats.test --decode_to_file=new_translation.txt

time python coms_test_script.py

time python bleu.py new_translation_fid.txt