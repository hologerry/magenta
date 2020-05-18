from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fontforge  # noqa
import os

# need python2, apt install python-fontforge


alphabet_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
valid_fonts_urls = open('svg_vae_data/glyphazzn_urls_valid.txt', 'r').readlines()
font_id_split_name = open('svg_vae_data/font_id_split_name.txt', 'w')
fonts_file_path = 'svg_vae_data/ttf_fonts'
sfd_path = 'svg_vae_data/sfd_font_glyphs'

font_id = 0
for font_line in valid_fonts_urls:
    font_name = font_line.strip().split(', ')[-1].split('/')[-1]
    split = font_line.strip().split(', ')[1]

    font_file_path = os.path.join(fonts_file_path, split, font_name)
    try:
        cur_font = fontforge.open(font_file_path)                # Open a font
    except Exception as e:
        print('Cannot open', font_name)
        print(e)
        continue

    try:
        # test all char exist
        # cur_font.selection.select(('ranges', None), 'A', 'Z')
        # cur_font.selection.select(('ranges', None), 'a', 'z')
        for char in alphabet_chars:
            cur_font.selection.select(char)
    except Exception as e:
        print(font_name, 'does not have all chars')
        print(e)
        continue

    font_id_split_name.write("{:06d}".format(font_id) + ', ' + split + ', ' + font_name + '\n')

    font_id += 1


font_id_split_name.close()
