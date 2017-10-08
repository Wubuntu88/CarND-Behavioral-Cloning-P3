# **Behavioral Cloning**

## Writeup Submission

###  This Document summarizes my work in the behavioural cloning project.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project files that relate to my neural network and data loading are all in the dev folder.

#### Project files
* dev/loader_generator.py - The file that loads the data, and provides the generator that will fetch training data on demand for the neural network.
* dev/nvidia_trainer_generator.py - The file that creates the nvidia neural network model and history object.
* dev/ztrain_nvidia_generator.py - The file that loads the data, sets up the neural network to the trained, and saves the model.
* dev/ztrain_prev_model.py - A file that will load a previously saved network, and retrain the network with new training data, and save that new network to a new file.

My project also contains several trained models.  The files with Seq in their names means that they were derived in sequence by training a pretrained network.  They are in the 'trained_models_sequence' folder.
#### The base network
* **nvidia_model_new_model_pretty_good.h5** - This is the base convolution neural network from which the other networks were trained in sequence.
#### Subsequently trained networks:
* **zTrainSeq01.h5** - The first network trained in sequence.
* **zTrainSeq01.h5** - The second network trained in sequence.
* **zTrainSeq01.h5** - The final network that looped the track without falling off.

My project also contains the following files provided by udacity:
* drive.py for driving the car in autonomous mode
* video.py for creating a video from the pictures that the drive.py creates.
* writeup_submission.md, summarizing the results (which is this document)

#### 2. Functional Code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py trained_models_sequence/zTrainSeq03.h5 videos/your_run
```
This will create a your pictures in the videos/your_run folder.  Note that the simulator must be running in autonomous mode and you must exit the simulator when you wish to stop the recording.

The .mp4 file can be generated using hte video.py file can be run with the following bash command:
```sh
python video.py videos/your_run
```
This command will save a video named your_run.mp4 in the same hierarchy as where the your_run directory lived (in the videos directory). 

#### 3. Description of code for usability, understandability, and readability.

__ztrain_nvidia_model.py__ - 
To train my network, one can use the ztrain_nvidia_model.py in the dev folder.  It loads the data, initializes the generators, trains the network, and saves the file.  
It loads the data from the '../zAggregateData/AllDataLocations/all_data.csv' file.  I did not include my data because it is massive and cannot go on github.  
The simulator saves data locations in the driving_log.csv file, so if one wanted to train a new model on new simulator data, one could save it in the '../zAggregateData/AllDataLocations/' location and add the 'driving_log.csv' to the path ending.
(Note that I did some tricks to load all data from various runs, which is why my final folder name is all_data.csv and not driving_log.csv).

The __train_nvidia_model.py__ file uses the **loader_generator.py** and **nvidia_trainer_generator** files.

**loader_generator.py** loads the data and initialize the generators.

**nvidia_trainer_generator** constructs the model object to be trained.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I experimented with three different architechtures: the initial regression architechture given by udacity, the lenet architecture, and the nvidia architecture.
I ended up using the nvidia architecture because it seemed to show the best results.

#### 2. Attempts to reduce overfitting in the model

To reduce over fitting, I performed two measures:
* collect lots of training data, and collect training data specifically on the locations that the car ran off the track on.
* I modified the nvidia architecture to include dropout layers after each convolutional layer or dense layer.

Both measures were effective.  Collecting data specifically on problem spots was very helpful.  The reason this step can be considered reducing overfitting is that the model had been trained without enough of the data on problem spots, so it was being trained in such a way that it did not account enough for this data to give accurate steering values.

Adding the dropout layers helped significantly in terms of the car staying on the road.  The training with dropout ended with validation loss that was higher than without dropout.  This is expected, because with some probability, the network will not update certain weights.  However, the car performed exceptionally better, staying in the middle of the road, not swerving, not running off the road as much.  I was quite shocked that using dropout had such a significant effect.  I used a drop probability of 50%.

#### 3. Model parameter tuning

The model used an adam optimizer, which adjusts the learning rate during training.

#### 4. Appropriate training data

I used many runs to create a large training data set.  Some runs had the car travel down the center of the road; and on some runs I hugged the side of the road.
In retrospect, I should have had many examples of the vehicle correcting, meaning it would start at a position almost off the road and correct to be at center.
I also collected good data on the specific difficult locations.  For example, I collected several short runs where the car avoids the dirt road.
More data was collected after the first model was built to correct some problems the model was having.  
This additional data specifically addressed more problem spots where the car was going off the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

###### Simple Neural Approach (Non-convolutional; kind of like regression)

My first step was to setup and run the simple neural network model provided by udacity to test it out and familiarize myself with the process.
The keras model is shown below:
```python
def train_reg(x_train, y_train):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        shuffle=True)
    return model, history
```
The model turned out to make the car swerve left and right too much.
This could be attributed to bad training data, but I did not have faith in a non-convolutional network, so I decided to quickly move on.
Here is a gif of the car in action:

-- Note: I give links to gifs because embedded gifs are too distracting, and these gifs in particular make one motion sick --

https://github.com/Wubuntu88/CarND-Behavioral-Cloning-P3/blob/master/videos/regression_model_run.gif

###### Simple Neural Approach (Non-convolutional; kind of like regression)

My second approach was to use the LeNet architecture shown in the udacity tutorial.
I tried it because it was recommended by udacity as being decent enough and I thought it would be a good place to start.
The model from the lenet performed fairly well.  It drove down the center of the road without significant swerving.
It did very well on the 'easy' parts, but failed on some of the hard parts, such as where the car can veer off onto the dirt road,
or when the car can drive into the side of the bridge.
In these cases, I did collect more data and train with more data, but I believe I was not focused enough in my data collection.
Instead of collect data on the problem spot, I would also collect more data on the easy parts.
This lead to having a bloated data set with not enough examples of the difficult driving locations.
I moved on to my new network; at the time I had not realized my mistake, and I wanted to try and see if a better architecture would make the difference.
If I had trained lenet on better data, perhaps I could have seen its true potential.
In any case, here is an example of lenet performing well:

https://github.com/Wubuntu88/CarND-Behavioral-Cloning-P3/blob/master/videos/lenet_run.gif



The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
