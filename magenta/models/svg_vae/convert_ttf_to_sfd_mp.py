from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fontforge  # need python2, apt install python-fontforge
import os
import multiprocessing as mp


alphabet_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
valid_fonts = open('svg_vae_data/font_id_split_name.txt', 'r').readlines()

fonts_file_path = 'svg_vae_data/ttf_fonts'
sfd_path = 'svg_vae_data/sfd_font_glyphs_mp'

process_nums = mp.cpu_count() - 2
lines_num = len(valid_fonts)
lines_num_per_process = lines_num // process_nums


def process(process_id, line_num_p_process):
    for i in range(process_id * line_num_p_process, (process_id + 1) * line_num_p_process):
        if i >= lines_num:
            break
        font_line = valid_fonts[i]
        font_id = font_line.strip().split(', ')[0]
        split = font_line.strip().split(', ')[1]
        font_name = font_line.strip().split(', ')[-1]

        font_file_path = os.path.join(fonts_file_path, split, font_name)
        try:
            cur_font = fontforge.open(font_file_path)                # Open a font
        except Exception as e:
            print('Cannot open', font_name)
            print(e)
            continue

        # already tried in single processs
        # try:
        #     # test all char exist
        #     # cur_font.selection.select(('ranges', None), 'A', 'Z')
        #     # cur_font.selection.select(('ranges', None), 'a', 'z')
        #     for char in alphabet_chars:
        #         cur_font.selection.select(char)
        # except Exception as e:
        #     print(font_name, 'does not have all chars')
        #     print(e)
        #     continue

        target_dir = os.path.join(sfd_path, split, "{}".format(font_id))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for char_id, char in enumerate(alphabet_chars):
            char_description = open(os.path.join(target_dir, '{}_{:02d}.txt'.format(font_id, char_id)), 'w')

            cur_font.selection.select(char)
            cur_font.copy()

            new_font_for_char = fontforge.font()
            new_font_for_char.selection.select(char)
            new_font_for_char.paste()
            new_font_for_char.fontname = "{}_".format(font_id) + font_name

            new_font_for_char.save(os.path.join(target_dir, '{}_{:02d}.sfd'.format(font_id, char_id)))

            char_description.write(str(ord(char)) + '\n')
            char_description.write(str(new_font_for_char[char].width) + '\n')
            char_description.write(str(new_font_for_char[char].vwidth) + '\n')
            char_description.write('{:02d}'.format(char_id) + '\n')
            char_description.write('{}'.format(font_id))

            char_description.close()

        cur_font.close()

processes = [mp.Process(target=process, args=(pid, lines_num_per_process)) for pid in range(process_nums+1)]

for p in processes:
    p.start()
for p in processes:
    p.join()

