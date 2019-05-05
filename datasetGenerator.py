import numpy as np
import cv2,os,pickle

cascade = cv2.CascadeClassifier('./haarCascades/frontalface_default.xml')


cam = cv2.VideoCapture(0)

i=0
offset=50
name=input('enter category: ')

datapath='dataSet'
imagesroot=os.path.join(datapath,'images')

while True:
    ret, img =cam.read()
    if not ret:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, 1.3, 5)
    facefound=False
    face=None
    for (x,y,w,h) in faces:
        facefound=True
        
        face=img.copy()[y-offset:y+h+offset,x-offset:x+w+offset]
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)
    waitkey=chr(cv2.waitKey(100) & 255)
    if waitkey=='q':
        break
    elif waitkey=='s' and facefound:
        i=i+1
        cv2.imwrite(imagesroot+"/face-"+name +'.'+ str(i) + ".jpg", face)
        
    if i>20:
        cam.release()
        break
cv2.destroyAllWindows()

def generate_labels_dict(datapath):
     imagesroot=os.path.join(datapath,'images')
     image_paths = [os.path.join(imagesroot, f) for f in os.listdir(imagesroot)]
     # images will contains face images
     images = []
     # labels will contains the label that is assigned to the image
     labels = []
     for image_path in image_paths:
         # Read the image and convert to grayscale
         #print(image_path)
         filetype=os.path.split(image_path)[-1].split(".")[-1]
         if filetype != 'jpg':
            continue
         image=cv2.imread(image_path)
         # Get the label of the image
         nbr = os.path.split(image_path)[-1].split(".")[0].replace("face-", "")
         #nbr=int(''.join(str(ord(c)) for c in nbr))
         print(nbr)
         # Detect the face in the image
         images.append(image)
         labels.append(nbr)
         cv2.imshow("Adding faces to traning set...", image)
         cv2.waitKey(10)
         # If face is detected, append the face to images and the label to labels
             
             
     # return the images list and labels list
     labels=set(labels)
     return labels
labelDict=generate_labels_dict(datapath)
print(labelDict)
pickle.dump(list(labelDict),open(os.path.join(datapath,'labelDict.pkl'),'wb'))