from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.utils import to_categorical
import cv2,os,pickle
import numpy as np


path = os.path.dirname(os.path.abspath(__file__))
dataPath = path+r'\dataSet'
trainingcycle=20

input_tensor = Input(shape=(128, 128, 3))  # this assumes K.image_data_format() == 'channels_last'
base_model = VGG19(input_tensor=input_tensor, include_top=False, weights='imagenet') 

def resizeAndPad(img, size=(128, 128)):
        h, w = img.shape[:2]

        sh, sw = size
        # interpolation method
        if h > sh or w > sw:  # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h

        # padding
        if aspect > 1: # horizontal image
            new_shape = list(img.shape)
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img=np.zeros(new_shape, dtype=np.uint8)
            h_offset=int((w-h)/2)
            new_img[h_offset:h_offset+h, :, :] = img.copy()

        elif aspect < 1: # vertical image
            new_shape = list(img.shape)
            new_shape[0] = h
            new_shape[1] = h
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape,dtype=np.uint8)
            w_offset = int((h-w) / 2)
            new_img[:, w_offset:w_offset + w, :] = img.copy()
        else:
            new_img = img.copy()
        # scale and pad
        scaled_img = cv2.resize(new_img, size, interpolation=interp)
        return scaled_img

def get_images_and_targets(datapath, labelDict, size=(128,128)):
    imagesroot=os.path.join(datapath,'images')
    image_paths = [os.path.join(imagesroot, f) for f in os.listdir(imagesroot)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        filetype=os.path.split(image_path)[-1].split(".")[-1]
        if filetype != 'jpg':
            continue
        image=cv2.imread(image_path)
        # Get the label of the image
        nbr = os.path.split(image_path)[-1].split(".")[0].replace("face-", "")
        #nbr=int(''.join(str(ord(c)) for c in nbr))
        print(nbr)
        # Detect the face in the image
        image=resizeAndPad(image,size)
        images.append(image)
        labels.append(labelDict.index(nbr))
        cv2.imshow("Adding faces to traning set...", image)
        cv2.waitKey(10)
        # If face is detected, append the face to images and the label to labels
            
            
    # return the images list and labels list
    targets=to_categorical(labels)
    print(targets)
    return np.array(images), np.array(targets)

datapath='dataSet'
labelDict=pickle.load(open(os.path.join(datapath,"labelDict.pkl"),'rb'))
images, targets=get_images_and_targets(datapath,labelDict,size=(128,128))
images=preprocess_input(images)

x=base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(200, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit(x=images,y=targets,epochs=trainingcycle)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:18]:
   layer.trainable = False
for layer in model.layers[18:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(x=images,y=targets,epochs=trainingcycle)

model.save('trainner/model.h5')