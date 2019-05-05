from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
import numpy as np
import cv2,os,pickle

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

cascade = cv2.CascadeClassifier('./haarCascades/frontalface_default.xml')
cam = cv2.VideoCapture(0)
model=load_model('trainner/model.h5')
datapath='dataSet'
labelDict=pickle.load(open(os.path.join(datapath,"labelDict.pkl"),'rb'))
labelImgs={}
for label in labelDict:
    labelImgs[label]=resizeAndPad(cv2.imread('targetEmoticons/'+label+".png"))
offset=50
while True:
    ret, img =cam.read()
    if not ret:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        if w*h==0 or y-offset<0 or x-offset<0:
            continue
        roi=img.copy()[y-offset:y+h+offset,x-offset:x+w+offset]
        face=resizeAndPad(roi.copy(),(128,128))
        predicted=model.predict(preprocess_input(np.array([face])))
        predicted=np.argmax(predicted)
        print(labelDict[predicted])
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('emoticon',labelImgs[labelDict[predicted]])

    cv2.imshow('img',img)
    if cv2.waitKey(10)=='q':
        break
cv2.destroyAllWindows()