 # Modern Style Transfer of Ancient Paintings

## Introduction

In the rich tapestry of art history, ancient paintings serve as a testament to bygone eras, encapsulating narratives that transcend time. "Modern Style Transfer of Ancient Paintings" is an ambitious endeavor to bridge the divide between historical artistry and contemporary design. This project innovates by selectively infusing modern art styles into segments of classical paintings, celebrating both ancient and avant-garde art.

## Motivation

While ancient art is revered, its relatability and interpretation within the modern cultural context are limited. This project reenvisions elements of classical paintings with contemporary designs, inviting a conversation that spans across the ages and reigniting the spirit of revolution in a language that speaks to today's audience.

## Problem Description

The challenge is to apply style transfer to select components of an image without altering the integrity of the entire painting. Our solution is a four-module system that individually addresses different aspects of the style transfer process, ensuring a nuanced and respectful enhancement of the original works.

## Modules

### Interaction Module
- Facilitates user interaction with an image.
- Captures click coordinates to feed a deep learning segmentation model.
- Generates a mask highlighting the selected region of interest.

### Semantic Segmentation Module
- Employs advanced deep-learning techniques.
- Discerns and delineates distinct elements within the paintings.

### Style Transfer Module
- Transforms user-selected segments stylistically.
- Adopts characteristics of modern art movements.
- Preserves the painting's untouched aspects.

### Reorganization Module
- Synthesizes style-transferred segments with the original.
- Ensures a seamless blend that respects the original composition.

## Getting Started

### Dependencies

- [PyTorch](http://pytorch.org/)

Optional dependencies:

- For CUDA backend:
  - CUDA 7.5 or above
- For cuDNN backend:
  - cuDNN v6 or above
- For ROCm backend:
  - ROCm 2.1 or above
- For MKL backend:
  - MKL 2019 or above
- For OpenMP backend:
  - OpenMP 5.0 or above

As for  style transfer module, After installing the dependencies, you'll need to run the following script to download the VGG model:

```
python models/download_models.py
```

This will download the original [VGG-19 model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md). The original [VGG-16 model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) will also be downloaded. By default the original VGG-19 model is used.

If you have a smaller memory GPU then using NIN Imagenet model will be better and gives slightly worse yet comparable results. You can get the details on the model from [BVLC Caffe ModelZoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). The NIN model is downloaded when you run the `download_models.py` script.

You can find detailed installation instructions for Ubuntu and Windows in the [installation guide](https://github.com/ProGamerGov/neural-style-pt/blob/master/INSTALL.md).

### Installation

#### 1.Segment Anything Installation

##### Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

##### <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

Additionally, masks can be generated for images from the command line:

```
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```


##### ONNX Export

SAM's lightweight mask decoder can be exported to ONNX format so that it can be run in any environment that supports ONNX runtime, such as in-browser as showcased in the [demo](https://segment-anything.com/demo). Export the model with

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

See the [example notebook](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb) for details on how to combine image preprocessing via SAM's backbone with mask prediction using the ONNX model. It is recommended to use the latest stable version of PyTorch for ONNX export.


##### <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)



#### 2. Style Transfer Installation

In the `transfer_style_module` file,  we present Windows Installation. You can more information about this module in original website: https://github.com/ProGamerGov/neural-style-pt/blob/72514050755394270abf526252cb9ff95c20dcf2/INSTALL.md.

If you wish to install PyTorch on Windows From Source or via Conda, you can find instructions on the PyTorch website: https://pytorch.org/

First, you will need to download Python 3 and install it: https://www.python.org/downloads/windows/. I recommend using the executable installer for the latest version of Python 3.

Then using https://pytorch.org/, get the correct pip command, paste it into the Command Prompt (CMD) and hit enter:


```
pip3 install torch===1.3.0 torchvision===0.4.1 -f https://download.pytorch.org/whl/torch_stable.html
```


After installing PyTorch, download the neural-style-pt Github respository and extract/unzip it to the desired location.

Then copy the file path to your style_transfer_module folder, and paste it into the Command Prompt, with `cd` in front of it and then hit enter.

In the example below, the style_transfer_module folder was placed on the desktop:

```
cd C:\Users\<User_Name>\Desktop\Modern Style Transfer of Ancient Paintings\style_transfer_module
```

You can now continue on to [installing neural-style-pt](https://github.com/ProGamerGov/neural-style-pt/blob/master/INSTALL.md#install-neural-style-pt), skipping the `git clone` step.

First we clone `neural-style-pt` from GitHub:

```
cd ~/
git clone https://github.com/Aph-xin/CV2023_Modern_Style_Transfer_of_Ancient_Paintings.git
cd transfer_style_module
```

Next we need to download the pretrained neural network models:

```
python models/download_models.py
```

You should now be able to run `neural-style-pt` in CPU mode like this:

```
python neural_style.py -gpu c -print_iter 1
```

If you installed PyTorch with support for CUDA, then should now be able to run `neural-style-pt` in GPU mode like this:

```
python neural_style.py -gpu 0 -print_iter 1
```

If you installed PyTorch with support for cuDNN, then you should now be able to run `neural-style-pt` with the `cudnn` backend like this:

```
python neural_style.py -gpu 0 -backend cudnn -print_iter 1
```

If everything is working properly you should see output like this:

```
Iteration 1 / 1000
  Content 1 loss: 1616196.125
  Style 1 loss: 29890.9980469
  Style 2 loss: 658038.625
  Style 3 loss: 145283.671875
  Style 4 loss: 11347409.0
  Style 5 loss: 563.368896484
  Total loss: 13797382.0
Iteration 2 / 1000
  Content 1 loss: 1616195.625
  Style 1 loss: 29890.9980469
  Style 2 loss: 658038.625
  Style 3 loss: 145283.671875
  Style 4 loss: 11347409.0
  Style 5 loss: 563.368896484
  Total loss: 13797382.0
Iteration 3 / 1000
  Content 1 loss: 1579918.25
  Style 1 loss: 29881.3164062
  Style 2 loss: 654351.75
  Style 3 loss: 144214.640625
  Style 4 loss: 11301945.0
  Style 5 loss: 562.733032227
  Total loss: 13711628.0
Iteration 4 / 1000
  Content 1 loss: 1460443.0
  Style 1 loss: 29849.7226562
  Style 2 loss: 643799.1875
  Style 3 loss: 140405.015625
  Style 4 loss: 10940431.0
  Style 5 loss: 553.507446289
  Total loss: 13217080.0
Iteration 5 / 1000
  Content 1 loss: 1298983.625
  Style 1 loss: 29734.8964844
  Style 2 loss: 604133.8125
  Style 3 loss: 125455.945312
  Style 4 loss: 8850759.0
  Style 5 loss: 526.118591309
  Total loss: 10912633.0
```



### Usage

In the notebook `Modern_Style_Transfer_of_Ancient_Paintings.ipynb`



Xxxxxx



