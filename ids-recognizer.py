import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from PIL import Image
import argparse
#import sys
import os
import subprocess
from subprocess import PIPE
import re
import json


parser = argparse.ArgumentParser(description='Detect Hanzi-components from image file and generate IDS if possible.')

parser.add_argument('image_files', nargs='*', help='Image file name to process')
parser.add_argument('--model', help='MLX-VLM model path', default='mlx-community/Qwen3.5-27B-heretic-8bit') 

args = parser.parse_args()


# Load the model
model_path = args.model
model_separator_pos = model_path.find('/')
model_name = model_path[model_separator_pos + 1:]

model, processor = load(model_path)
config = load_config(model_path)

prompt = '''<image> Locate every component of the Kanji.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/upper-left/upper-right/lower-left/lower-right/enclosed/middle)
'''

component_prompt = '''<image> Run OCR for component (or structure) of the Kanji and output the result.
'''

character_prompt = '''<image> Run Kanji (or traditional Hanzi) OCR and output the result.
'''

def run_VLM (images, prompt):
    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images = len(images)
    )
    
    # Generate output
    response = generate(model, processor, formatted_prompt, images,
                        max_tokens = 512, temperature=0.0,
                        verbose=False)
    print (response)
    return response.text

def detect_ids (X1, Y1, X2, Y2, Component_Text, Component_Position):
    number_of_components = len(Component_Text)
    if number_of_components == 2:
        if Component_Text[1] == '四点底':
            Component_Text[1] = '灬'
        match Component_Position[0]:
            case 'left':
                if ( ( Component_Position[1] == 'right' ) or
                     ( Component_Position[1] == 'full-surround' ) or
                     ( Component_Position[1] == 'full' ) ):
                    return f'⿰{Component_Text[0]}{Component_Text[1]}'

            case 'upper-left':
                if Component_Position[1] == 'lower-right':
                    if ( ( Component_Text[0] == '雨' ) or
                         ( Component_Text[0] == '⺮' ) or
                         ( Component_Text[1] == '儿' ) or
                         ( Component_Text[1] == '女' ) or
                         ( Component_Text[1] == '心' ) or
                         ( Component_Text[1] == '虫' ) or
                         ( Y2[0] <= Y1[1] ) ):
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'
                elif Component_Position[1] == 'lower-left':
                    if ( Component_Text[0] == '辶' ):
                        return f'⿺{Component_Text[1]}{Component_Text[0]}'
                    else:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                # elif Component_Position[1] == 'upper-right':
                #     return f'⿰{Component_Text[0]}{Component_Text[1]}'
                elif Component_Text[1] == '口':
                    return f'⿸{Component_Text[0]}{Component_Text[1]}'
                elif Component_Text[0] == '𠂇':
                    return f'⿸{Component_Text[0]}{Component_Text[1]}'
                elif Component_Text[0] == '麻':
                    if ( Y2[0] > Y1[1] ):
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                else:
                    return f'⿰{Component_Text[0]}{Component_Text[1]}'

            case 'lower-left':
                if Component_Position[1] == 'lower-right':
                    if {Component_Text[0]} == '攵':
                        return f'⿰{Component_Text[0]}{Component_Text[1]}'
                    else:
                    # if ( ( Component_Position[1] == 'upper-right' ) or
                    #      ( Component_Position[1] == 'enclosed'    ) )
                    #     if ( ( {Component_Text[0]} == '辶' ) or
                    #          ( {Component_Text[0]} == '廴' ) or
                    #          ( {Component_Text[0]} == '走' ) ):
                        return f'⿺{Component_Text[0]}{Component_Text[1]}'
                elif ( ( Component_Position[1] == 'upper-right' ) or
                       ( Component_Position[1] == 'full-surround' ) or
                       ( Component_Position[1] == 'full' ) or
                       ( Component_Position[1] == 'enclosed' ) ):
                    return f'⿺{Component_Text[0]}{Component_Text[1]}'

            case 'right':
                if Component_Position[1] == 'left':
                    return f'⿰{Component_Text[1]}{Component_Text[0]}'

            case 'above':
                if ( ( Component_Position[1] == 'below' ) or
                     ( Component_Position[1] == 'full-surround' ) ):
                    return f'⿱{Component_Text[0]}{Component_Text[1]}'

            case 'upper':
                if ( ( Component_Position[1] == 'below' ) or
                     ( Component_Position[1] == 'lower' ) or
                     ( Component_Position[1] == 'full-surround' ) ):
                    if Component_Text[0] == '气':
                        return f'⿹{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'

            case 'below':
                if Component_Position[1] == 'above':
                    return f'⿱{Component_Text[1]}{Component_Text[0]}'
                elif Component_Position[1] == 'surround-from-above':
                    return f'⿵{Component_Text[0]}{Component_Text[1]}'

            case 'upper-right':
                if Component_Position[1] == 'lower-left':
                    if Component_Text[0] == '戈':
                        return f'⿹{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                elif Component_Position[1] == 'enclosed':
                    return f'⿱{Component_Text[0]}{Component_Text[1]}'

            case 'full-surround':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) ):
                    if Component_Text[0] == '凵':
                        return f'⿶{Component_Text[0]}{Component_Text[1]}'
                    elif Component_Text[0] == '几':
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '广' ) or
                           ( Component_Text[0] == '麻' ) ):
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '門' ) or
                           ( Component_Text[0] == '冂' ) ):
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿴{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-above':
                if ( ( Component_Position[1] == 'below' ) or
                     ( Component_Position[1] == 'covered' ) or
                     ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'full-surround' ) ):
                    if ( ( Component_Text[0] == '虍' ) or
                         ( Component_Text[0] == '鹿' ) or
                         ( Component_Text[0] == '广' ) ):
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '宀' ) or
                           ( Component_Text[0] == '穴' ) or
                           ( Component_Text[0] == '冖' ) or
                           ( Component_Text[0] == '雨' ) ):
                        return f'⿱{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-below':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) ):
                    if Component_Text[0] == '皿':
                        return f'⿱{Component_Text[1]}{Component_Text[0]}'
                    else:
                        return f'⿶{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-left':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'covered' ) ):
                    if Component_Text[0] == '疒':
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'
                    elif ( ( Component_Text[0] == '門') or
                           ( Component_Text[0] == '冂') ):
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿷{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-upper-left':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'lower-right' ) ):
                    if Component_Text[0] == '匚':
                        return f'⿷{Component_Text[0]}{Component_Text[1]}'
                    elif Component_Text[0] == '几':
                        return f'⿵{Component_Text[0]}{Component_Text[1]}'
                    else:
                        return f'⿸{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-lower-left':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'upper-right' ) ):
                    return f'⿺{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-right':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) ):
                    return f'⿼{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-upper-right':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'lower-left' ) ):
                    return f'⿹{Component_Text[0]}{Component_Text[1]}'

            case 'surround-from-lower-right':
                if ( ( Component_Position[1] == 'middle' ) or
                     ( Component_Position[1] == 'enclosed' ) or
                     ( Component_Position[1] == 'upper-left' ) ):
                    return f'⿽{Component_Text[0]}{Component_Text[1]}'

    elif number_of_components == 3:
        if ( ( ( Component_Position[0] == 'upper' ) or
               ( Component_Position[0] == 'above' ) )
             and
             ( Component_Position[1] == 'middle' )
             and
             ( ( Component_Position[2] == 'lower' ) or
               ( Component_Position[2] == 'below' ) ) ):
            if Component_Position[1] == Component_Position[2]:
                return f'⿱{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
            else:
                return f'⿳{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'
        elif ( ( ( Component_Position[0] == 'above' ) or
                 ( Component_Position[0] == 'upper' ) )
               and
               ( Component_Position[1] == 'lower-left' )
               and
               ( Component_Position[2] == 'lower-right' ) ):
            return f'⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}'
        elif ( ( ( Component_Position[0] == 'above' ) or
                 ( Component_Position[0] == 'upper' ) )
               and
               ( Component_Position[1] == 'surround-from-above' )
               and
               ( Component_Position[2] == 'middle' ) ):
            if Component_Text[1] == '囗':
                return f'⿱{Component_Text[0]}⿴{Component_Text[1]}{Component_Text[2]}'
            else:
                return f'⿱{Component_Text[0]}⿵{Component_Text[1]}{Component_Text[2]}'
        elif ( ( Component_Position[0] == 'upper-left' )
               and
               ( Component_Position[1] == 'upper-right' )
               and
               ( ( Component_Position[2] == 'lower' ) or
                 ( Component_Position[2] == 'below' ) or
                 ( ( Component_Position[2] == 'lower-right' ) and
                   ( X1[2] < X2[0] ) )
                ) ):
            return f'⿱⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}'
        elif ( ( Component_Position[0] == 'left' )
               and
               ( Component_Position[1] == 'upper-right' )
               and
               ( Component_Position[2] == 'lower-right' ) ):
            return f'⿰{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
        elif ( ( Component_Position[0] == 'left' )
               and
               ( Component_Position[1] == 'upper-right' )
               and
               ( Component_Position[2] == 'lower-right' ) ):
            return f'⿰{Component_Text[0]}⿱{Component_Text[1]}{Component_Text[2]}'
        elif ( ( Component_Position[0] == 'upper-left' )
               and
               ( Component_Position[1] == 'full-surround' )
               and
               ( Component_Position[2] == 'enclosed' ) ):
            if Component_Text[1] == '門':
                return f'⿰{Component_Text[0]}⿵{Component_Text[1]}{Component_Text[2]}'
            else:
                return f'⿰{Component_Text[0]}⿴{Component_Text[1]}{Component_Text[2]}'
    elif number_of_components == 4:
        if ( ( ( Component_Position[0] == 'upper' ) or
               ( Component_Position[0] == 'above' ) )
             and
             ( Component_Position[1] == 'lower-left' )
             and
             ( Component_Position[2] == 'lower-right' )
             and
             ( Component_Position[3] == 'below' ) ):
            if ( ( Component_Text[0] == Component_Text[1] ) and
                 ( Component_Text[1] == Component_Text[2] ) ):
                return f'⿱⿱{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
            else:
                return f'⿳{Component_Text[0]}⿰{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
        elif ( ( Component_Position[0] == 'upper-left' )
               and
               ( Component_Position[1] == 'upper-right' )
               and
               ( Component_Position[2] == 'surround-from-above' )
               and
               ( Component_Position[3] == 'enclosed' ) ):
            if ( Component_Text[2] == '冖' ):
                return f'⿱⿱⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'
            else:
                return f'⿳⿰{Component_Text[0]}{Component_Text[1]}{Component_Text[2]}{Component_Text[3]}'


def run_OCR_for_glyph_image (image_file, prompt, TSV_OUTPUT_PATH, OUTPUT_PATH):
    im = Image.open(image_file)
    image_width, image_height = im.size
    basename = os.path.splitext(os.path.basename(image_file))[0]

    ids_file_name  = f'{OUTPUT_PATH}/{basename}_ids.txt'
    full_file_name = f'{OUTPUT_PATH}/{basename}_full.txt'
    if os.path.isfile(ids_file_name):
        print( f'{ids_file_name} already exists.')
        ids_destfile = open (ids_file_name, 'r', encoding = 'utf-8')
        ids = ids_destfile.read()
        ids_destfile.close()
        return ids
    elif os.path.isfile(full_file_name):
        print( f'{full_file_name} already exists.')
        full_destfile = open (full_file_name, 'r', encoding = 'utf-8')
        full = full_destfile.read()
        full_destfile.close()
        return full
    else:
        print (image_file, prompt)
        images = [ image_file ]

        response = run_VLM (images, prompt)

        component_number = 0
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        Component_Text = []
        Component_Position = []
        with open(f'{TSV_OUTPUT_PATH}/{basename}.tsv', 'w', encoding = 'utf-8') as tsv_destfile:
            for line_match in re.findall('(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\n?', response):
                x1, y1, x2, y2, line_text, position = line_match
                x1 = int (x1)
                y1 = int (y1)
                x2 = int (x2)
                y2 = int (y2)
                orx1 = round ( ( x1 * image_width  ) / 1000 )
                ory1 = round ( ( y1 * image_height ) / 1000 )
                orx2 = round ( ( x2 * image_width  ) / 1000 )
                ory2 = round ( ( y2 * image_height ) / 1000 )
                # orw  = round ( ( ( x2 - x1 ) * image_width)  / 1000 )
                # orh  = round ( ( ( y2 - y1 ) * image_height) / 1000 )
                X1.append(orx1)
                Y1.append(ory1)
                X2.append(orx2)
                Y2.append(ory2)
                Component_Text.append(line_text)
                Component_Position.append(position)
                print (f'{orx1}	{ory1}	{orx2}	{ory2}	{line_text}	{position}')
                print (f'{orx1}	{ory1}	{orx2}	{ory2}	{line_text}	{position}',
                       file=tsv_destfile)
                component_number = component_number + 1
                if ( ( (orx2 - orx1) > 0 ) and
                     ( (ory2 - ory1) > 0 ) ):
                    im_crop = im.crop((orx1, ory1, orx2, ory2))
                    im_crop.save(f'{TSV_OUTPUT_PATH}/{basename}_comp{component_number}.png')

        if len(Component_Text) == 3:
            if ( ( ( Component_Position[0] == 'upper-right' ) and
                   ( Component_Position[1] == 'full-surround' ) ) or
                 ( ( Component_Position[0] == 'upper-left' ) and
                   ( Component_Position[1] == 'upper-right' ) ) ):
                if ( ( abs(X1[1] - X1[0]) < 5 ) and
                     ( abs(X2[1] - X2[0]) < 5 ) and
                     ( Y1[1] <= Y2[0] ) ):
                    if X1[1] < X1[0]:
                        X1[0] = X1[1]
                    if Y1[1] < Y1[0]:
                        Y1[0] = Y1[1]
                    if X2[0] < X2[1]:
                        X2[0] = X2[1]
                    if Y2[0] < Y2[1]:
                        Y2[0] = Y2[1]
                    del X1[1]
                    del Y1[1]
                    del X2[1]
                    del Y2[1]
                    orig_comp2 = Component_Text[1]
                    del Component_Text[1]
                    del Component_Position[1]
                    print(f'-> ({X1[0]},{Y1[0]})-({X2[0]},{Y2[0]})')
                    if ( ( (X2[0] - X1[0]) > 0 ) and
                         ( (Y2[0] - Y1[0]) > 0 ) ):
                        im_crop = im.crop((X1[0], Y1[0], X2[0], Y2[0]))
                        comp1_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp1.png'
                        im_crop.save(comp1_image_file_name)
                        comp1_response = run_VLM ([comp1_image_file_name], component_prompt)
                        print (f'new component = "{comp1_response}".')
                        if len(comp1_response) == 1:
                            Component_Text[0] = comp1_response
                        else:
                            Component_Text[0] = f'⿱{Component_Text[0]}{orig_comp2}'
                    else:
                        Component_Text[0] = f'⿱{Component_Text[0]}{orig_comp2}'
                    comp2_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp2.png'
                    if (os.path.isfile(comp2_image_file_name)):
                        os.remove(comp2_image_file_name)
                    comp3_image_file_name = f'{TSV_OUTPUT_PATH}/{basename}_comp3.png'
                    if (os.path.isfile(comp3_image_file_name)):
                        os.rename(comp3_image_file_name, comp2_image_file_name)

        ids = detect_ids(X1, Y1, X2, Y2, Component_Text, Component_Position)
        if ids:
            with open(f'{OUTPUT_PATH}/{basename}_ids.txt',
                      'w', encoding = 'utf-8') as ids_destfile:
                print(ids, file=ids_destfile)
        elif len(Component_Text) == 1:
            char_response = run_VLM (images, character_prompt)
            print (f'character = "{char_response}".')
            if (os.path.isfile(full_file_name)):
                os.remove(full_file_name)
            if len(char_response) == 1:
                with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                    print(char_response, file=full_destfile)
            elif Component_Position[0] == 'full-surround':
                with open(full_file_name, 'w', encoding = 'utf-8') as full_destfile:
                    print(Component_Text[0], file=full_destfile)

        with open(f'{OUTPUT_PATH}/{basename}.txt', 'w', encoding = 'utf-8') as destfile:
            destfile.write(response)

        with open(f'{OUTPUT_PATH}/{basename}.prompt', 'w', encoding = 'utf-8') as prompt_file:
            prompt_file.write(prompt)
        return ids


proc = subprocess.run("ipfs add -- | cut -d' ' -f2", shell=True, input=prompt, stdout=PIPE, stderr=PIPE, text=True)
IPFS_CID = proc.stdout.rstrip('\r\n')

print (IPFS_CID)

OUTPUT_PATH = f'{model_name}/{IPFS_CID}/tsv_pct100'
os.makedirs(OUTPUT_PATH, exist_ok=True)
TSV_OUTPUT_PATH = f'{model_name}/{IPFS_CID}/tsv_pct100'
os.makedirs(TSV_OUTPUT_PATH, exist_ok=True)

print (OUTPUT_PATH, TSV_OUTPUT_PATH)

for image_file_name in args.image_files:
    ids = run_OCR_for_glyph_image (image_file_name, prompt,
                                   TSV_OUTPUT_PATH, OUTPUT_PATH)
    print (f'{image_file_name} : {ids}\n')
