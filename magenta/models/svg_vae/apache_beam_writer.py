import apache_beam as beam
import os
import pyarrow
glyph_list = []
fontdirs = []
'''
{'uni': int64,  # unicode value of this glyph
'width': int64,  # width of this glyph's viewport (provided by fontforge)
'vwidth': int64,  # vertical width of this glyph's viewport
'sfd': binary/str,  # glyph, converted to .sfd format, with a single SplineSet
'id': binary/str,  # id of this glyph
'binary_fp': binary/str}  # font identifier (provided in glyphazzn_urls.txt)
'''
for root, dirs, files in os.walk('svgvae_need_format/train/'):
   for dir in dirs:
       fontdirs.append(dir)
fontdirs.sort()
for footdir in fontdirs:
   cur_path = 'svgvae_need_format/train/' + footdir + '/'
   for idx in range(1, 52 + 1):
       f = open(cur_path + '/' + footdir + '-' + f'{idx:02d}' + '.txt')
       lines = f.read().split('\n')
       g_idx_dict = {}
       g_idx_dict['uni'] = int(lines[0])
       g_idx_dict['width'] = int(lines[1])
       g_idx_dict['vwidth'] = int(lines[2])
       g_idx_dict['id'] = lines[3]
       f_sfd = open(cur_path + '/' + footdir + '-' + f'{idx:02d}' + '.sfd', mode='rb')
       g_idx_dict['sfd'] = f_sfd.read()
       g_idx_dict['binary_fp'] = lines[4]
       # print(g_idx_dict)
       glyph_list.append(g_idx_dict)
       f.close()
       f_sfd.close()

with beam.Pipeline() as p:
   records = p | 'Read' >> beam.Create(glyph_list)
   _ = records | 'Write' >> beam.io.WriteToParquet('glyphs-parquetio',
       pyarrow.schema(
           [('uni', pyarrow.int64()), ('width', pyarrow.int64()), ('vwidth', pyarrow.int64()),
            ('sfd', pyarrow.string()), ('id', pyarrow.string()), ('binary_fp', pyarrow.string())]
       )
   )

