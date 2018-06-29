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

## Results

Once you run the test_network, this is what you get :

![ironman](https://user-images.githubusercontent.com/19779081/42071726-6bbb13fe-7b7a-11e8-87f5-5b7f6b9cb33d.PNG)

Isn't it cool? Your own Iron Man Detector :D

Dataset has been kept small so that everyone can train the neural network on their on systems. 
