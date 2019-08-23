from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem
class CodeComment(text_problems.Text2TextProblem):

    @property
    def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def approx_vocab_size(self):
        """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
        return 2**14  # ~16k

    @property
    def max_subtoken_length(self):
        """Maximum subtoken length when generating vocab.

        SubwordTextEncoder vocabulary building is quadratic-time wrt this variable,
        setting it to None uses the length of the longest token in the corpus.

        Returns:
            an integer or None
        """
        return 25

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        # del data_dir
        # del tmp_dir
        # del dataset_split
        print('***************************************', dataset_split, '****************************************************************************')

        if dataset_split=='eval':
            dataset_split = 'val'

        tdats_fn = tmp_dir + '/new_tdats.'+ dataset_split
        coms_fn = tmp_dir + '/new_coms.'+ dataset_split

        with open(tdats_fn, 'r') as tdats_file, open(coms_fn, 'r') as coms_file:
            for tdats_line, coms_line in zip(tdats_file, coms_file):
                tdats_line = tdats_line.strip()
                coms_line = coms_line.strip()
                tdats_line = tdats_line.split(', ')[1]
                coms_line = coms_line. split(', ')[1]
                yield {
                    "inputs": tdats_line,
                    "targets": coms_line,
                }