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

[nvidia_network]: ./dev/model.png "Model Visualization"

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

###### LeNet Model Approach

My second approach was to use the LeNet architecture shown in the udacity tutorial.
This is how the LeNet Model is in keras:
```python
def train_lenet(x_train, y_train, nb_epoch=10):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(layer=Cropping2D(cropping=((70, 20), (0, 0))))
    # lenet architecture
    # Convolution #1
    model.add(layer=Convolution2D(nb_filter=6,
                                  nb_row=5,
                                  nb_col=5,
                                  activation='relu'))
    # Pooling #1
    model.add(layer=MaxPooling2D())

    # Convolution #2
    model.add(layer=Convolution2D(nb_filter=6,
                                  nb_row=5,
                                  nb_col=5,
                                  activation='relu'))
    # Pooling #2
    model.add(layer=MaxPooling2D())

    # Flatten into fully connected layers
    model.add(layer=Flatten())
    model.add(layer=Dense(120))
    model.add(layer=Dense(84))
    model.add(layer=Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        shuffle=True,
                        batch_size=256,
                        nb_epoch=nb_epoch)
    return model, history
```

I tried this model because it was recommended by udacity as being decent enough and I thought it would be a good place to start.
The model from the lenet performed fairly well.  The car drove down the center of the road without swerving.
It did very well on the 'easy' parts, but failed on some of the hard parts, such as where the car can veer off onto the dirt road,
or when the car can drive into the side of the bridge.
In these cases, I did collect more data and train with more data, but I believe I was not focused enough in my data collection.
Instead of collect data on the problem spot, I would also collect more data on the easy parts.
This lead to having a bloated data set with not enough examples of the difficult driving locations.
I moved on to my new network; at the time I had not realized my mistake, and I wanted to try and see if a better architecture would make the difference.
If I had trained lenet on better data, perhaps I could have seen its true potential.
In any case, here is an example of lenet performing well:

https://github.com/Wubuntu88/CarND-Behavioral-Cloning-P3/blob/master/videos/lenet_run.gif

And a case where it drives off the road:

https://github.com/Wubuntu88/CarND-Behavioral-Cloning-P3/blob/master/videos/lenet_offroad.gif

**Augmenting training data:**
In this section, I augmented the training data in two ways:
* I flipped the Image and flipped the sign of the steering value.
* I added the two side cameras and modified the steering value with a correction.

###### Nvidia Architecture Approach

I used the Nvidia Architecture because Udacity and a member on the forums recommended it.
The Nvidia architecture looks like the following in keras:
```python
def train_nvidia(train_generator, num_train_samples,
                 validation_generator, num_validation_samples,
                 nb_epoch=10, batch_size=32):
    """
    This method is used to get the model to then train.  The history object of the model is also returned.
    :param train_generator: The generator that fetches the next training data batch.
    :param num_train_samples: The number of training samples that the generator will eventually fetch.
    :param validation_generator: The generator that fetches the next validation data batch.
    :param num_validation_samples: The number of validation data that the validation generator will eventually fetch.
    :param nb_epoch: The number of epochs that the model will train for.
    :param batch_size: The batch size of each training mini run.
    :return: a tuple of 2 items containing (model, history object)
    """
    dropout_p = 0.5
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(layer=Cropping2D(cropping=((70, 20), (0, 0))))
    # lenet architecture
    # Convolution #1
    model.add(layer=Convolution2D(nb_filter=24,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2, 2),
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Convolution #2
    model.add(layer=Convolution2D(nb_filter=36,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2, 2),
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Convolution #3
    model.add(layer=Convolution2D(nb_filter=48,
                                  nb_row=5,
                                  nb_col=5,
                                  subsample=(2, 2),
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Convolution #4
    model.add(layer=Convolution2D(nb_filter=64,
                                  nb_row=3,
                                  nb_col=3,
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Convolution #5
    model.add(layer=Convolution2D(nb_filter=64,
                                  nb_row=3,
                                  nb_col=3,
                                  activation='relu'))
    model.add(Dropout(dropout_p))
    # Flatten into fully connected layers
    model.add(layer=Flatten())
    model.add(layer=Dense(100))
    model.add(Dropout(dropout_p))
    model.add(layer=Dense(50))
    model.add(Dropout(dropout_p))
    model.add(layer=Dense(10))
    model.add(Dropout(dropout_p))
    model.add(layer=Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(generator=train_generator,
                                  samples_per_epoch=num_train_samples,
                                  validation_data=validation_generator,
                                  nb_val_samples=num_validation_samples,
                                  nb_epoch=nb_epoch)
    return model, history
```

The model performed well, but still had some of the same problems that LeNet did.  Also, It drove in a swerving manner, even on the 'easy' parts.

**Addition of Dropout**

One addition I made to the architecture was dropout.  I got this idea because someone had mentioned it on the forums.
Adding dropout made my model perform much better, much to my surprise.  The car Drove much more smoothly, and did not swerve off the road on the 'easy' parts.

#### 2. Final Model Architecture

My final model architecture was the nvidia model with dropout.
I will discuss two data preprocessing steps, and show a dispay of the network.

**Data Pre-processing**

I performed two preprocessing techniques that are visible in the first two layers of the neural network diagram.

1) I mean rescaled the data such that it had a range of \[-0.5, 0.5].
The pixel values had initially been \[0,255].  This rescaling helps the neural network converge on a solution faster.
2) I cropped the image such that the top 70 pixels, and the bottom 20 pixels were ignored.  The top area shows information about the sky and trees, and the bottom area mostly is the hood of the car.
These portions of the image do not provide helpful information at best, and mislead the neural network into picking up patterns that do not really influence steering direction.
Thus, the picture data fed into the neural network is mostly road-specific.

Both of these steps were done in the Lambda layer of the neural network.  This layer allows us to modify the data without having to run a batch preprocessing job on the raw data.

I also had read from some individuals on the forums that further reshaped their image to be 64x64.
They claimed this significantly reduced the training time, without degrading the performance.
I chose not to do this for severals reasons.
The first of which is that I thought it would be troublesome, and I did not know of a way to do it with a Lambda layer.
Also, my training was slow, but not unbearably slow.
I was using an NVIDIA GTX 750 Ti, which completed my training in anywhere from 5 to 15 minutes, depending on the number of epochs.

Here is a diagram of the final modified NVIDIA network:

![alt text][nvidia_network]

#### 3. Creation of the Training Set & Training Process

I captured many runs of training data.  I collected around 15 recordings.
This figure shows
Some were long, including several laps around the course.  Several were short, being 10 seconds or so.
A few recording included driving in the center, two had the car hugging each side of the road.
My training process usually involved training on a given set of records, noticing the failings, and just getting more data.
This did not bring that much improvement, and dramatically increase my training time.
I began creating multiple short recording of problems spots.  
The helped significantly, and helped me overcome cases where the car would make errors repeatedly.
Multiple runs of the same configuration and training data sometimes yielded better or worse models.
I got several models that were good enough, and saved them for later reference.

I became frustrated with the inconsistency of the neural network training, so I decided to switch my approach.
Instead of starting from scratch every time, I decided I would choose a 'good' model that I had saved and retrain it on select data of problem spots.

I chose the 'nvidia_model_new_model_pretty_good.h5' in the trained_models_sequence directory because it performed well.
I figured out how to load a model, train it on new data, and save a new model so the old one was perserved.

The code for training a previously trained network in in dev/ztrain_prev_model.py.  I list the code here:
```python
from dev.loader_generator import load_data_samples, generator
from sklearn.model_selection import train_test_split
from keras.models import load_model

training_file_name = '../MyDatazTurnHardCorner03/driving_log.csv'
load_model_path = '../trained_models_sequence/zTrainSeq02.h5'
save_model_path = '../trained_models_sequence/zTrainSeq03.h5'

samples = load_data_samples(path_to_csv_file=training_file_name)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

model = load_model(load_model_path)

batch_size = 64
train_generator = generator(train_samples, batch_size=batch_size, side_cameras=False)
validation_generator = generator(validation_samples, batch_size=batch_size, side_cameras=False)


model.fit_generator(generator=train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=10)

model.save(save_model_path)
```

I ended up training the original model an additional three times.  Each time I built on the previously trained model.
I used the same trouble spot training data for all three runs.
For some reason, the network had significant difficulty learning this step.
The location I am referring to is the curve right after the dirt road.
However, despite its intransigence, it finally learned to stay on the road during this section - but just barely.
My final model is the zTrainSeq03.h5 in the trained_models_sequence directory.

**A note on training and validation loss**
The training error vs validation loss comparison did not help me determine if the model was overfitting.
They were both quite similar.  In fact, the loss did not really help me at all.
A low loss in either the training or validation sets did not give me an indication of how well it would perform on the road.

**The video**
Here is a link to where the video of the full lap of driving is:

**GIF:**
https://github.com/Wubuntu88/CarND-Behavioral-Cloning-P3/blob/master/videos/nvidia_model_run.gif

**MP4:**
https://github.com/Wubuntu88/CarND-Behavioral-Cloning-P3/blob/master/videos/nvidia_model_run.mp4

(note: if you download the video in firefox, it may say that the mimetype is not supported; the download worked in Safari)
