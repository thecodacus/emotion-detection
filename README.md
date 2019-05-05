# Facial Expression Recognition
Detect Emotion Using Deeplearning
## Some experiments with transfer learning with keras deep learning models
<img src="https://github.com/thecodacus/emotion-detection/raw/master/results/MLFacialExpressionResult.gif" width="400">

## How to use
I made this project structured almost in similar to my [Face Recognition](https://github.com/thecodacus/Face-Recognition) Project,

### Prerequisites
* Python 3.6
* OpenCV 3.4.1
* Numpy
* Keras

### Installing
* clone the repo
* Create an empty Folder named "dataSet" in the same directory where the python scripts are
* Create an empty folder called trainer In same directory

### Running the tests
* run the datasetGenerator.py and enter the face expression that you want to add to your dataset.
* press "S" once you are ready to save/capture the expression, keep capturing save expression in different lighting and background
* press "Q" to quit or it will automatically exit after 20 samples
once you have saved a set of samples for one expression, repeat the process for more expression 

* run trainer.py. it till take some time to complete the training wait until it completes 
* run detector.py

## Authors

* **[Anirban Kar](http://thecodacus.com/author/admin/)**
