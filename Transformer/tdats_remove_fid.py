tmp_dir = 'tmp_dir'
dataset_split = 'test'
tdats_fn = tmp_dir+'/tdats.'+ dataset_split
new_tdats_fn = tmp_dir+'/new_tdats.'+ dataset_split

with open(tdats_fn, 'r') as tdats_file, open(new_tdats_fn, 'w') as new_tdats_file:
    for tdats_line in tdats_file:
        tdats_line = tdats_line.strip()
        tdats_line = tdats_line.split(', ')[1]
        new_tdats_file.write(tdats_line+'\n')