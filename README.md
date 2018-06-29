# IronManDetector

It identifies whether or not an image is of Iron Man. 

## Dependencies

1. Python
2. Keras
3. Tensorflow
4. Matplotlib

## Dataset

The dataset is prepared by downloading 366 iron man images from google. The file urls.txt is received on running the javascript file.
The images then can be downloaded using download_images.py 
Here in this code, I have already downloaded the images and created images directory containing both ironman and not_ironman images. 
Not iron man images are obtained by randomly sampling 366 images that do not contain Santa from the UKBench dataset, a collection of ~10,000 images used for building and evaluating Content-based Image Retrieval (CBIR) systems

## Setup

To train the model, run train_network.py using following command
python train_network.py --dataset images --model ironman_not_ironman.model
The model once trained is saved as ironman_not_ironman.model and can be used directly later without having the need to train again and again

To test the model, run test_network.py using following command
python test_network.py --model ironman_not_ironman.model --input_img examples/ironman.jpg

Keep the directory structure same as that of mine so as to get the code running smoothly.



