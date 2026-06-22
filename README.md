# IDS-Recognizer
This program is a automatic IDS (Ideographic Description Sequence)
generator.  It recognizes components and structure of Chinese
characters contained in each input image using Visual Language Model
(VLM) and automatically generates IDS corresponding to each image.

This program currently supports only
[MLX-VLM](https://github.com/Blaizzy/mlx-vlm) package for the
[MLX](https://github.com/ml-explore/mlx) framework for Apple silicon
Mac.


## Preparation

Please install MLX-VLM package.
```
% uv pip install -U mlx-vlm
```


## Usage
```
% uv run ids-recognizer.py glyph-image-file.png

% uv run ids-recognizer.py --model mlx-community/diffusiongemma-26B-A4B-it-8bit *.jpg
```
