from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fontforge  # need python2, apt install python-fontforge
import os


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
        cur_font.selection.select(('ranges', None), 'A', 'Z')
        cur_font.selection.select(('ranges', None), 'a', 'z')
    except Exception as e:
        print(font_name, 'does not have all chars')
        print(e)
        continue

    font_id_split_name.write("{:06d}".format(font_id) + ', ' + split + ', ' + font_name + '\n')

    target_dir = os.path.join(sfd_path, split, "{:06d}".format(font_id))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for char_id, char in enumerate(alphabet_chars):
        char_description = open(os.path.join(target_dir, '{:06d}_{:02d}.txt'.format(font_id, char_id)), 'w')

        cur_font.selection.select(char)
        cur_font.copy()

        new_font_for_char = fontforge.font()
        new_font_for_char.selection.select(char)
        new_font_for_char.paste()
        new_font_for_char.fontname = "{:06d}_".format(font_id) + font_name

        new_font_for_char.save(os.path.join(target_dir, '{:06d}_{:02d}.sfd'.format(font_id, char_id)))

        char_description.write(str(ord(char)) + '\n')
        char_description.write(str(new_font_for_char[char].width) + '\n')
        char_description.write(str(new_font_for_char[char].vwidth) + '\n')
        char_description.write('{:02d}'.format(char_id) + '\n')
        char_description.write('{:06d}'.format(font_id))

        char_description.close()

    cur_font.close()
    font_id += 1

valid_fonts_urls.close()
font_id_split_name.close()
