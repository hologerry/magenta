import random

org_split_file = 'svg_vae_data/font_id_split_name.txt'
tgt_split_file = 'svg_vae_data/font_id_split_name_eval.txt'

org_lines = open(org_split_file, 'r').readlines()
with open(tgt_split_file, 'w') as f:
    for line in org_lines:
        r = random.random()
        if r < 0.1 and 'train' in line:
            line = line.replace('train,', 'eval,')
        f.write(line)
