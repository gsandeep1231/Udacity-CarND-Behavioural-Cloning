
##Introduction
This is the Behaviour Cloning project Udacity's Self Driving Nanodegree program. 
I have explained below my process of behaviour cloning to train the car to drive around a simulated track.

##Collection of Data
Data was collected by combining the training data received from Udacity and by using the Simulator. 
In Simulator's Training mode, I drove the car couple of laps around the track to record proper driving data. 
I also needed recovery data for training since just the correct driving data was not sufficient. This recovery data
is needed to let the model know what to do when it goes away from the track. I drove the car to the side of the track
and only recorded the data where it would correct itself to come back to the center of the track. This was done 
wherever the model needed more training, especially at tight corners.

##Pre-processing the data
Pre-processing the data was very important since there is a lot of unwanted noise in the data. We want to focus only
on the track. So I trimmed the data of the background and converted the image to thresholded grayscale image so 
that it is easy and faster for the model to train rather than training on raw color image.
Below is the pipeline I used:
