tmp_dir = 'tmp_dir'

dataset_split = 'test'
coms_fn = tmp_dir+'/coms.'+ dataset_split
new_trans_fn = 'new_translation.txt'
fid_trans_fn = 'new_translation_fid.txt'

with open(coms_fn, 'r') as coms_file, open(new_trans_fn, 'r') as new_trans_file, open(fid_trans_fn, 'w') as fid_trans_file:
	for coms_line, new_trans_line in zip(coms_file, new_trans_file):
		coms_line = coms_line.strip()
		new_trans_line = new_trans_line.strip()
		coms_line = coms_line. split(', ')[0]
		trans_line_list = new_trans_line.split()
		if len(trans_line_list)>13:
			trans_line_list[12] = trans_line_list[len(trans_line_list)-1]
			trans_line_list = trans_line_list[:13]
		if len(trans_line_list)<13:
			for i in range(13-len(trans_line_list)):
				trans_line_list.append('<NULL>')
		fid_trans_file.write(coms_line+'\t'+' '.join(trans_line_list)+'\n')
