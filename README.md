# DS5220-Final-Project

## Important Work to be Done:

### 0. See what this is creating

Blocked by lack of model

### 1. Create Batch Function

Blocked by lack of model

### 2. Make sure data functions work

So I fine tuned the data functions. I used your codes as a template for what I needed to and translated it to pandas/numpy. However I didn't do a couple things cause they did not make sense.

1. attach_annotations(imgData, annData) you go through each subset of captions based on indices. It seems like an unecessary operation if our goal is just to get the mapped captions to be attached to the image metadata and the caption that they corresspond to no?

2. You create a matrix for your captions. So image 1 will have a matrix column that has five rows. I didn't see the point of that either. PANDAS has great searching capabilities. While now the train_data-1.pkle file might be a bit bigger than your file, we are able to keep to a OK PANDAS format (I personally have never seen matrices as cell values, if you have you can make this argument to me and I'll push a fix.)

All changes can be found in 

### 3. Create Grid Search for CV

Blocked by lack of model

### 4. Save Analysis that we need

### 5. Run it

### 6. BLEU score integration

Currently have code that works. Will need to check if it still works once we can get captions loaded.

### 7. COCO API

Not really sure what we were looking at for this.

### 8. Potentionally considering switching to PyTorch

PyTorch seems like a simpler framework to work with, but we will see.


