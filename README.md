# Repository for Low-Light Enhancement of Focused Light Fields
The code in this repository is designed to denoise low-light focused light field images.

In particular, it covers the following networks:
* ENH - A simple U-Net architecture designed to not overcompress microlens images
* ENH-S - Passes a stack of microlens images or a patched version of the light field to ENH
* ENH-W - Passes a stack of microlens images or a patched version of the light field through two U-Nets, aka a W-Net. 

In all cases, post training full light fields could be run at once on the GPU, avoiding the need for any post-processing in terms of stitching.

Several methods are also available in this repository for learning depth from microlens images.

## Data Location
Data for the low-light light field dataset can be found at:

All light fields are captured with a Raytrix R5

## Usage

TODO

## Example Results

TODO

## Our System
All of the above was tested on the following system:
* Titan X (Maxwell) GPU
* Ubuntu 16.04
* Python 3.6
* PyTorch 1.3.0
