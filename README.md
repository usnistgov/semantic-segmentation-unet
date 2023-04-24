# SemanticSegmentation

Semantic Segmentation PyTorch model.

This repo needs a submodule from: https://github.com/usnistgov/pytorch_utils

Submodule management documentation can be found at https://git-scm.com/book/en/v2/Git-Tools-Submodules

# Input Data Constraints

Input data assumptions:
- image type: N channel image with one of these pixel types: uint8, uint16, int32, float32
- mask type: grayscale image with one of these pixel types: uint8, uint16, int32
- masks must be integer values of the class each pixel belongs to
- mask pixel value 0 indicates background/no-class
- each input image must have a corresponding mask with the same filename (but potentially different file extension) 
- each image/mask pair must be identical size
