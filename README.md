# Eye Gaze Tracking

In this project, we shall create a system for eye-gaze tracking. The system uses deep learning and image recognition to predict the eye-gaze direction.


## Overview
Eye tracking is becoming a very important capability across many domains, including security, psychology, computer vision, and medical diagnosis. Also, gaze is important for security applications to analyze suspicious gaze behavior. A use case in educational institutes is the automated analysis of the student’s eye gazes during an examination to help minimize malpractices.
In this project, we're going to implement a deep learning solution for eye gaze tracking on images and video streams, and ultimately deploying the solution in real-time settings.

## Dataset
For the dataset, we're going to use https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/, to generate synthetic images of eye regions along with their labels (the direction of where the eye is looking). This data generation method combines a novel generative 3D model of the human eye region with a real-time rendering framework. The model is based on high-resolution 3D face scans and uses real-time approximations for complex eyeball materials and structures as well as anatomically inspired procedural geometry methods for eyelid animation. The training set of UnityEyes captures a large degree of appearance variation, which enables us to test against challenging images.

![Alt text](./images/unityeye.png?raw=true "synthetic data using generative 3D eye region model" )

## Model
Using efficient-Net, we’re predicting the direction of the gaze
vector, predicting the coordinates on the out- put image. Af-
ter that, we’re using OpenCV to get the eye pupil’s position
and other eye’s landmarks, then connecting it to the coordi-
nates of the gaze vector to draw the gaze vector.
The current architecture uses EfficientNet replacing the out-
put layers with 2 fully connected layers to match the intended
output size (2 in our case, the 𝑥 and 𝑦 positions of the gaze
vector).
![Alt text](./images/prediction.png?raw=true "predicted gaze-direction" )
## Usage
- Install the required dependencies:

```
pip install -r requirements.txt
```
- Run the inference script using the following command:
```
python facial_landmarks.py
```