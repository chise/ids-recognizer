import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from PIL import Image
import sys
import os
import subprocess
from subprocess import PIPE
import re
import json

# Load the model
model_path = "mlx-community/diffusiongemma-26B-A4B-it-8bit"
model_name = "diffusiongemma-26B-A4B-it-8bit"
model, processor = load(model_path)
config = load_config(model_path)

prompt = '''<image> Locate every components of the Chinese character.
Report each component with bbox coordinates as TSV format like:
X0	Y0	X1	Y1	component	position (above/bellow/left/right/surrounding/covered/overlapped)
'''

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)


def run_OCR_for_glyph_image (image_file, prompt, TSV_OUTPUT_PATH, OUTPUT_PATH):
    im = Image.open(image_file)
    image_width, image_height = im.size
    basename = os.path.splitext(os.path.basename(image_file))[0]

    print (image_file, prompt)
    image = [ image_file ]

    # Generate output
    response = generate(model, processor, formatted_prompt, image,
                        max_tokens = 4096, temperature=0.0,
                        verbose=False)
    print (response, type(response))
    response = response.text

    print (response)
    component_number = 1
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
            orw  = round ( ( ( x2 - x1 ) * image_width)  / 1000 )
            orh  = round ( ( ( y2 - y1 ) * image_height) / 1000 )
            print (f'{orx1}	{ory1}	{orx2}	{ory2}	{line_text}	{position}')
            print (f'{orw}	{orh}	{orx1}	{ory1}	{line_text}	{position}',
                   file=tsv_destfile)
            im_crop = im.crop((orx1, ory1, orx2, ory2))
            im_crop.save(f'{TSV_OUTPUT_PATH}/{basename}_comp{component_number}.png')
            component_number = component_number + 1

        with open(f'{OUTPUT_PATH}/{basename}.txt', 'w', encoding = 'utf-8') as destfile:
            destfile.write(response)

        with open(f'{OUTPUT_PATH}/{basename}.prompt', 'w', encoding = 'utf-8') as prompt_file:
            prompt_file.write(prompt)


image_file_name = sys.argv[1]

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
