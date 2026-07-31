import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

def run_VLM (images, prompt, model, processor, config):
    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images = len(images)
    )
    
    # Generate output
    response = generate(model, processor, formatted_prompt, images,
                        max_tokens = 1024, temperature=0.0,
                        verbose=False)
    print (response)
    return response.text
