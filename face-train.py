import pickle
import cv2

import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'image')

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# recognizer : 
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_id = {}
y_laybels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
			path = os.path.join(root,file)
			label = os.path.basename(root).replace(' ', '-').lower()
			#print(label,'   ',path)

			if not label in label_id:
				label_id[label] = current_id
				current_id +=1
			id_ = label_id[label] 
			#print('labels_id  = ', label_id)

			pil_image = Image.open(path).convert("L") # grayscale
			size = (550,550)
			final_mage = pil_image.resize(size,Image.ANTIALIAS)

			image_array = np.array(pil_image,'uint8')
			#print(image_array)

			faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
		
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_laybels.append(id_)


# pickle : 

with open('labels.pickle','wb') as f :
	pickle.dump(label_id,f)

recognizer.train(x_train,np.array(y_laybels))
recognizer.save('trainner.yml')
print('Finish')