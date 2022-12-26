# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf
import ntpath 
ntpath.realpath = ntpath.abspath

model = tf.keras.models.load_model('keras_model.h5')
# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
    status , frame = camera.read()
    if status:
        frame = cv2.flip(frame , 1)
        resize_frame = cv2.resize(frame, (224,224))
        
        resize_frame = np.expand_dims(resize, axis = 0)
        resize_frame = resize_frame/255.0
        prediction = mymodel.predict(resize_frame)
        
	rock = int(prediction[0][0]*100)
	paper = int(prediction[0][1]*100)
        scissor= int(prediction[0][2]*100)
	
	print(f"Rock : {rock} %, Paper : {paper} %, Scissor : {scissor} %") 
	
        cv2.imshow('feed' , frame)
        code = cv2.waitKey(1)
        
        if code == 32:
            break
			

		
		
		
# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
