# P3-BehavioralCloning


## Synopsis

In Project #3 - Behavior Cloning, we were asked to train a neural network that would enable a car to drive itself around a simulated track. Using a driving simulator provided by the amazing people at Udacity, we could use the "Training Mode" and drive around a track, simultaneously collecting images taken by 3 cameras around the car, as well as driving specifications. These images are stored locally, and the paths to each set of images, along with the driving specifications at the moment the picture was taken, are stored in a spreadsheet called driving_log.csv. These specifications include speed, steering angle, and throttle. With this information, one should be able to train a neural network to associate each image with a specific steering angle. Once this model is compiled, we could activate the "Autonomous Mode" in the simulator, and by running the model, the car should be able to continuously drive around the track. We were able to complete this project by using a number of augmentation techniques and experimenting with different types of models to achieve the ultimate goal of driving a full lap without crossing the lines of the track.

## Requirements

### Packages & Libraries

- [Python 3.5] (https://www.continuum.io/downloads)
- [TensorFlow] (https://www.tensorflow.org/versions/)
- [Keras] (https://keras.io/)
- [numpy] (https://anaconda.org/anaconda/numpy)
- [flask-socketio] (https://anaconda.org/pypi/flask-socketio)
- [eventlet] (https://anaconda.org/conda-forge/eventlet)
- [pillow] (https://anaconda.org/anaconda/pillow)
- [h5py] (https://anaconda.org/conda-forge/h5py)
- [OpenCV] (https://anaconda.org/menpo/opencv3)

### Simulator: 
- [Linux] (https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
- [macOS] (https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
- [Windows32] (https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
- [Windows64] (https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

### Other:

- [Training data] (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
- [Original drive.py] (https://d17h27t6h515a5.cloudfront.net/topher/2017/January/586c4a66_drive/drive.py)

## Project Structure
- `model.json`- Model architecture
- `model.py` - The script used to create and train the model.
- `model.h5` - Model weights
- `drive.py`- The script to drive the car
- `driving_log.csv` - Each row in this sheet correlates to an image with the steering angle, throttle, brake, and speed of the car
- `IMG` - This folder contains the images taken while driving. Images are taken from the front, left, and right cameras on the car

## Data Collection and Analysis

### Data Collection

We were asked to first drive around a simulated track and record data. While in the "Training Mode", the simulator collects images taken from the center, left, and right cameras of the car and stores them. The simulator tracks the steering angle, speed, and throttle of the car at each moment that an image is captured, and stores these values with the paths to the corresponding images in a spreadsheet called driving_log.csv. When we decide that we have done enough training, we can stop our recording and observe the results. 

The idea now is to create a model that can predict what steering angle the car should take based on the image it sees from the front facing camera. In other words, with an input of an image the model should output a steering angle the car should take. This model can be achieved by collecting our own data by driving around the track for many laps. If you decide to record your own data, make sure to cover the following two things:

1. Recovery: You should collect lots of data that covers your car driving off of the track, then returning back to the track, at as many spots as possible. In the event that your car does drive off the track when it is driving itself, it will be able to recognize how to recover and get back on the road. You may think it is a good idea to train your model on 50 laps of data where you have perfectly driven in the middle of the road continously, but in reality if your model falters at any point and you drive off of the road, your car will have no refernece for how to recover. 

2. Turning from different angles: When collecting data at the turns on the track, make sure to approach each turn at a different angle  each time. Of course you want your car to perfectly drive down the center of the lane for each turn, but realistically your car may be swerving as it approaches this turn and if it swivles into a turn along the leftmost part of the lane, it will not be familiar with what it is seeing, as it has only made this turn from the center.

We spent a good amount of time collecting our own data but after many failed attempts at creating a functional model, we decided to use Udacity's provided data, as it did a better job of covering these turning angles, as well as recovery, across the track. The data contains 24,108 images, with 8,036 taken from the center, left, and right cameras each.

### Data Analysis

The images produced by the simulator are in `.jpg` format and its dimensions are `160 x 320 x 3`. Below are three examples of images taken from the left, center, and right cameras, respectively:

![alt tag](https://s23.postimg.org/b8qy3p7bf/rsz_left_pic.png)


![alt tag](https://s23.postimg.org/mwkzy8wgb/rsz_center_pic.png)


![alt tag](https://s23.postimg.org/qewzumxcb/rsz_right_pic.png)


### Cropping
One improvement we made to the images was cropping 20% from the top and bottom of the image. I did this to help the neural network train more efficiently. By cutting out the sky/trees and the hood of the car, it would be focusing as much as possible on the features on the road. Below, you can see the center image from above image with the sky and hood of the car cropped out:

![alt tag](https://s23.postimg.org/qiqtaw2tn/rsz_cropped2.png)


### Resizing the image
By making the image smaller, we reduced the number of features the model needs to interpret, thereby making it easier to process the image. I reduced mine to `64 x 64` to make it about the size of a thumbnail. Below, you can see a resized version of the previous image:

![alt tag](https://s23.postimg.org/fkfjspe8b/rsz_resized64.png)

Note: These two steps were done in both model.py and drive.py, to make sure the images fed to the model during testing were consistent with the images it was trained on. This is very important, as the model cannot be expected to drive the car autonomously with images being fed to it that are unrepresentative of the images it received during training.


## Augmentation

As mentioned before, we were provided with 8,036 images from each camera to train our model on and while this may seem like a lot, it proves to be insufficient when you want to train an effective model. The key to designing a good neural network is to train it with as much data as you possibly can. You could choose to drive around the track for hours and keep collecting data until you think you have enough for your model to train on. The other option is to augment the data that is provided. We can assume that the data provided to us should be enough to train our model, so using a few augmentation techniques, we can turn our data into more data! Here are a few methods I used to augment my existing data and produce enough to train my model:

1. Flipping Images: This was a simple technique that resulted in doubling my data. During training, while selecting my data, I would randomly flip the image and negate the steering angle. For each image, I now had a mirrored image for the model to train on with opposite steering angle. Although this step may seem pointless, it accomplishes the goal of increasing data for the model to train on. I kept this random, but made sure to only flip non-zero images, since it would not be efficient to do this for a 0 image.

2. Shifting Images: The car will be at different positions on the track than it was trained on, so it is important to generate data that will allow the car to recognize many different positions for a given image. Thus, we randomly shift each image horizontally to simulate this effect, and vertically to simulate moving up or down a slope. We also add a 0.002 steering angle unit/pixel to shift right, and subtract to shift the car left. 

3. Using left, right images: They're given to us for a reason! Although initially we may assume we can train our model using just our center images, there is a clear way to increase our data by simply utilizing the left and right images taken by the car. Keep in mind that we are given a single steering angle for each image, so to utilize the left and right images, we need to shift adjust the steering angle accordingly. By subtracting -0.25 from the steering angle for images taken from the right camera, while adding 0.25 to the steering angle taken from the left camera, we can train our model to adjust itself when it sees images in the center camera that may have been taken from the left or right camera. This step not only increases our data, but actually adds an aspect of recovery since it selfs adjusts by shifting back to the center if it is seeing an image it saw from a side camera.

### Removing Extreme Data

The most useful step in augmenting the data was removing data that would bias the model too far to the right or left. We provided the model with a threshold value which would decrease incrementally with each epoch. The model would reject values that were less than this threshold value, so as the epochs got higher and higher, my model would be accepting smaller and smaller values. If the angle was higher than 0.1, we were refer to the threshold to see if this angle was acceptable. If it was higher than 0.1, it would be rejected. This was an extremeley important step, as it allowed the car to drive around the track without being exposed steering angles higher than 0.1, making it take less drastic turns and swerve less on the road. 

## Training the Model

For our model, we utilized a keras fit_generator model. This trains the model on data that is given batch-by-batch. The advantage of using the fit_generator is that it lets you augment your data on your CPU while training your model in parallel on the GPU. The generator keeps taking batches until it has received enough, so it is easy to feed it copious amounts of data. We randomly removed 10% of the training data to use for validation data. We generated a fit_generator for both the training and validation data, and gave both to our model. Our model accepted 10 samples of values for validation. The resulting validation loss, along with the actual performance along the track, would be the final determining factors of our model's fitness.

### Model Structure

Below is the basic structure of the model used in this project:

![alt_tag](https://s23.postimg.org/a8aot91yz/Screen_Shot_2017_01_10_at_7_34_40_PM.jpg)

### Model Specifications
The model was built using keras and used an Adam Optimizer with 1e-4 learning rate. We originally tried using the default Adam optimizer provided by Keras, but received better results when given the option of controlling the learning rate. The model is trained on 9 epochs with 50,000 samples per epoch, with a batch size of 250. Loss is computed as `mean_squared_error` and we used `mean_absoulte_error` as our metric. The advantage of using `mean_absolute_error` is that it gives a real representation of how far off your predicted values are from the targeted values. This was helpful in this project, as we were able to see by how much the steering angle was varying with each epoch of each model.


## Performance
Our model performed very well, as the car was able to continuously drive around the track for hours with our final model. The final validation loss was 0.0437. The final `mean_absolute_error` of validation was 0.1685. Both of these values were very close to the final training values, which lead us to believe that our model was well equipped in predicting correct steering angles for the track.

![alt tag](https://s30.postimg.org/pigaqkt8x/Screen_Shot_2017_01_10_at_11_12_17_PM.jpg)


To run the model, activate the simulator and run:

`python drive.py model.py`

You can see a demo of my model [here] (https://youtu.be/_0kk6iRglls)

## Reflection
This project helped bring to light some of the fundamentals of machine learning. Specifically, the need for a large amount of data. The more data your model is provided, the better it can understand and predict accurate output. By increasing my samples/epoch from 20,000 to 50,000, we were able to weave around turns we were unable to make before. Data augmentation was the truly the most valuable lesson, as we were able to take a small amount of data and produce countless versinos of it that covered a multitude of scenarios that our car could encounter. 

## Acknowledments
Special thanks to Tanuj Thapliyal for helping me build my new computer, fully equiped with an NVIDIA Titan X, Vivek Yadav for inspiring myself, as well as most of our Udacity classmates, with his many augmentation techniques, Brok Bucholtz for telling me when it's time to abandon everything I've done and start fresh, and to all my classmates on Slack who help us preserve our sanity and learn together.
