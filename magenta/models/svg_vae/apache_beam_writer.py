import os
import pickle
# import sys

import apache_beam as beam
import pyarrow

# Write apache beam parquetio
'''
{'uni': int64,  # unicode value of this glyph
'width': int64,  # width of this glyph's viewport (provided by fontforge)
'vwidth': int64,  # vertical width of this glyph's viewport
'sfd': binary/str,  # glyph, converted to .sfd format, with a single SplineSet
'id': binary/str,  # id of this glyph
'binary_fp': binary/str}  # font identifier (provided in glyphazzn_urls.txt)
'''
cur_split = 'train'
glyph_list_path = f"svg_vae_data/glyph_list_{cur_split}.pkl"
target_beam_parquetio_file_prefix = f'svg_vae_data/glyphs-parquetio-{cur_split}/glyphs-parquetio'
glyph_list = []
fontdirs = []

sfd_dir = 'svg_vae_data/sfd_font_glyphs_mp'
font_id_split_names = open('svg_vae_data/font_id_split_name.txt', 'r').readlines()

if not os.path.exists(glyph_list_path):
    print("Processing sfd files ...")
    for id_split_name in font_id_split_names:
        font_id = id_split_name.strip().split(', ')[0]
        split = id_split_name.strip().split(', ')[1]

        if split != cur_split:
            continue

        cur_font_path = os.path.join(sfd_dir, split, font_id)
        for char_id in range(52):
            char_des = open(os.path.join(cur_font_path, '{}_{:02d}.txt'.format(font_id, char_id)), 'r')
            lines = char_des.readlines()
            g_idx_dict = {}
            g_idx_dict['uni'] = int(lines[0].strip())
            g_idx_dict['width'] = int(lines[1].strip())
            g_idx_dict['vwidth'] = int(lines[2].strip())
            g_idx_dict['id'] = lines[3].strip()

            f_sfd = open(os.path.join(cur_font_path, '{}_{:02d}.sfd'.format(font_id, char_id)), mode='rb')
            g_idx_dict['sfd'] = f_sfd.read()
            g_idx_dict['binary_fp'] = lines[4].strip()
            # print(g_idx_dict)
            glyph_list.append(g_idx_dict)

            char_des.close()
            f_sfd.close()

    with open(glyph_list_path, 'wb') as f:
        pickle.dump(glyph_list, f)
    print("Processed all font files")
else:
    with open(glyph_list_path, 'rb') as f:
        glyph_list = pickle.load(f)
    print("Loaded processed font files")

print('Submitting to beam ...')

with beam.Pipeline() as p:
    records = p | 'Read' >> beam.Create(glyph_list)
    _ = records | 'Write' >> beam.io.WriteToParquet(target_beam_parquetio_file_prefix,
                                                    pyarrow.schema([('uni', pyarrow.int64()), ('width', pyarrow.int64()), ('vwidth', pyarrow.int64()),
                                                                    ('sfd', pyarrow.string()), ('id', pyarrow.string()), ('binary_fp', pyarrow.string())])
                                                    )
