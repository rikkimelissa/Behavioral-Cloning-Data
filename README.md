# **Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Image"
[image2]: ./examples/recovery1.jpg "Recovery Image 1"
[image3]: ./examples/recovery1.jpg "Recovery Image 2"
[image4]: ./examples/flipped.png "Flipped Image"
[image5]: ./examples/sat_image.jpg "Sat Image 1"
[image6]: ./examples/sat_image2.jpg "Sat Image 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_weights_final.h5 containing the trained weights for a convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 and run2.mp4 of the car driving at 7mph and 20mph respectively

#### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_weights_final.h5
```

#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 6 and 120 (model.py lines 58-101). The model uses image input in both the RGB and HSV color spaces.

The model includes RELU and softmax layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. I also cropped out the top 50 and lower 20 pixels of each image to avoid training on meaningless information. 

To augment the data, I reversed the images and measurements and added these to the training/validation data.

#### Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. I also used a smaller network to reduce the chances of overfitting. I also augmented the data by flipping images and using them as training data.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road at various points in the track, focused bridge driving and recovery bridge driving, and focused driving on the turn after the bridge.

### Model Architecture and Training Strategy

#### Solution Design Approach

The overall strategy for deriving a model architecture was to use the simplest one that worked. My first step was to use a convolution neural network model similar to the LeNet architecture. I also played around a bit with the Nvidia architecture, but had issues with running out of memory. I tried using generators with this architecture but found that training was extremely slow. Since my first network worked pretty well on most of the road, I instead focused efforts on improving this network.

My first attempt with this network design worked pretty well but always failed on a specific turn after the bridge. To combat this, I added in the saturation channel of the image in HSV colorspace, which helped the model differentiate between black road and brown dirt sides. I also added more activation layers to make the neural net less linear. I also realized after many training iterations that the model.fit function did not shuffle the data prior to splitting into training and validation sets as I had assumed. Shuffling the data before the split helped immensely.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Before shuffling the data appropriately, I found that my model had really low training loss but not so great validation loss because it was overfitting on regular driving and validating on the recovery data which was added later. After shuffling, the model had really low training and validation losses.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road up to about 20mph.

#### Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
- 5x5 convolution layer
- RELU
- Max pooling
- 5x5 convolution layer
- RELU
- Max pooling
- Dropout with .5 probability
- Fully connected layer 120 outputs
- RELU
- Fully connected layer 40 outputs
- Softmax
- Output

I used 3 merged mini models to form this architecture. The first model handled RGB images for the first 6 layers. The second model handled S images for the first 6 layers. The third and final model merged these outputs and added on the last 6 layers.

#### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded most of one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct on over steering. I added recovery data at three points along the road, on the bridge, and on the curve after the bridge. These images show what a recovery looks like starting from the left side:

![alt text][image2]
![alt text][image3]


To augment the data set, I also flipped images and angles thinking that this would help the neural net generalize. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image4]

To augment the data set, I also added as input the saturation channel for the image in HSV color space. Here are two examples of how the image has high contrast in saturation between road and off-road.

![alt text][image5]
![alt text][image6]

After the collection process, I had 4,860 images that I shuffled and split into 60% for training and 40% for validation after shuffling. I preprocessed the data by cropping the top 50 and lower 20 pixels. I also normalized the pixel values. I used an adam optimizer so that manually training the learning rate wasn't necessary. I found 3 epochs to be a good number for low loss but not overfitting.

#### Final results

Watch the video!
[![Watch the video][image1]](https://youtu.be/sZWqiCoFFnI)


