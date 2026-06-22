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

parser.add_argument('image_file_name', help='Image file name to process') 
parser.add_argument('--model', help='MLX-VLM model path', default='mlx-community/Qwen3.5-27B-heretic-8bit') 

args = parser.parse_args()


# Load the model
model_path = args.model
model_separator_pos = model_path.find('/')
model_name = model_path[model_separator_pos + 1:]

model, processor = load(model_path)
config = load_config(model_path)

prompt = '''<image> Locate every components of the Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/below/left/right/full-surround/surround-from-above/surround-from-below/surround-from-left/surround-from-right/surround-from-upper-left/surround-from-upper-right/surround-from-lower-left/surround-from-lower-right/covered/middle/overlapped)
'''

def run_VLM (images, prompt):
    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images = len(images)
    )
    
    # Generate output
    response = generate(model, processor, formatted_prompt, images,
                        max_tokens = 4096, temperature=0.0,
                        verbose=False)
    print (response, type(response))
    return response.text

def detect_ids (X1, Y1, X2, Y2, Component_Text, Component_Position):
    number_of_components = len(Component_Text)
    if number_of_components == 2:
        if ( ( Component_Position[0] == 'left' ) and
             ( Component_Position[1] == 'right' ) ):
            return f'⿰{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'above' ) and
               ( Component_Position[1] == 'below' ) ):
            return f'⿱{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'full-surround' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿴{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'surround-from-above' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿵{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'surround-from-below' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿶{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'surround-from-left' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿷{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'surround-from-upper-left' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿸{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'surround-from-upper-right' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿹{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'surround-from-lower-left' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿺{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'surround-from-right' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿼{Component_Text[0]}{Component_Text[1]}'
        elif ( ( Component_Position[0] == 'surround-from-lower-right' ) and
               ( ( Component_Position[1] == 'middle' ) or
                 ( Component_Position[1] == 'covered' ) ) ):
            return f'⿽{Component_Text[0]}{Component_Text[1]}'

def run_OCR_for_glyph_image (image_file, prompt, TSV_OUTPUT_PATH, OUTPUT_PATH):
    im = Image.open(image_file)
    image_width, image_height = im.size
    basename = os.path.splitext(os.path.basename(image_file))[0]

    print (image_file, prompt)
    images = [ image_file ]

    response = run_VLM (images, prompt)

    print (response)
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
            im_crop = im.crop((orx1, ory1, orx2, ory2))
            component_number = component_number + 1
            im_crop.save(f'{TSV_OUTPUT_PATH}/{basename}_comp{component_number}.png')

        ids = detect_ids(X1, Y1, X2, Y2, Component_Text, Component_Position)
        if ids:
            with open(f'{OUTPUT_PATH}/{basename}_ids.txt', 'w', encoding = 'utf-8') as ids_destfile:
                print(ids, file=ids_destfile)

        with open(f'{OUTPUT_PATH}/{basename}.txt', 'w', encoding = 'utf-8') as destfile:
            destfile.write(response)

        with open(f'{OUTPUT_PATH}/{basename}.prompt', 'w', encoding = 'utf-8') as prompt_file:
            prompt_file.write(prompt)


#image_file_name = sys.argv[1]
image_file_name = args.image_file_name

print (image_file_name)

proc = subprocess.run("ipfs add -- | cut -d' ' -f2", shell=True, input=prompt, stdout=PIPE, stderr=PIPE, text=True)
IPFS_CID = proc.stdout.rstrip('\r\n')

print (IPFS_CID)

OUTPUT_PATH = f'{model_name}/{IPFS_CID}/tsv_pct100'
os.makedirs(OUTPUT_PATH, exist_ok=True)
TSV_OUTPUT_PATH = f'{model_name}/{IPFS_CID}/tsv_pct100'
os.makedirs(TSV_OUTPUT_PATH, exist_ok=True)

print (OUTPUT_PATH, TSV_OUTPUT_PATH)

run_OCR_for_glyph_image (image_file_name, prompt, TSV_OUTPUT_PATH, OUTPUT_PATH)
