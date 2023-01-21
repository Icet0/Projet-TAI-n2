import base64
import os

import pandas as pd
import numpy as np
import scipy.io
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS


import time

from PIL import Image
import mtcnn
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import Model, Sequential
# from keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image as imageKeras



#-----------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    #preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    #Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img

def loadVggFaceModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
  #GlobalAveragePooling

	model.add(Activation('softmax'))
	
	#you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
	from tensorflow.keras.models import model_from_json
	model.load_weights('weights/vgg_face_weights.h5')
	
	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
	
	return vgg_face_descriptor

model = loadVggFaceModel()
print("vgg face model loaded")

#------------------------
exists = os.path.isfile('representations.pkl')

if exists != True: #initializations lasts almost 1 hour. but it can be run once.
	
	#download imdb data set here: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ . Faces only version (7 GB)
	mat = scipy.io.loadmat('imdb_data_set/imdb.mat')
	print("imdb.mat meta data file loaded")
	columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score", "celeb_names", "celeb_id"]

	instances = mat['wiki'][0][0][0].shape[1]

	df = pd.DataFrame(index = range(0,instances), columns = columns)

	for i in mat:
		if i == "wiki":
			current_array = mat[i][0][0]
			for j in range(len(current_array)):
				#print(j,". ",columns[j],": ",current_array[j][0])
				df[columns[j]] = pd.DataFrame(current_array[j][0])

	print("data frame loaded (",df.shape,")")

	#-------------------------------

	#remove pictures does not include any face
	df = df[df['face_score'] != -np.inf]

	#some pictures include more than one face, remove them
	df = df[df['second_face_score'].isna()]

	#discard inclear ones
	df = df[df['face_score'] >= 5]

	#-------------------------------
	#some speed up tricks. this is not a must.

	#discard old photos
	df = df[df['photo_taken'] >= 2000]

	print("some instances ignored (",df.shape,")")
	#-------------------------------

	def extractNames(name):
		return name[0] if (len(name) > 0) else None

	
	df['celebrity_name'] = df['name'].apply(extractNames)
	def getImagePixels(image_path):
		return cv2.imread("imdb_data_set/%s" % image_path[0]) #pixel values in scale of 0-255

	tic = time.time()
	df['pixels'] = df['full_path'].apply(getImagePixels)
	toc = time.time()

	print("reading pixels completed in ",toc-tic," seconds...") #3.4 seconds
	detector = MTCNN()

	def findFaceRepresentation(img):
		try:
			border_rel = 0 # increase or decrease zoom on image
			detections = detector.detect_faces(img)
			x1, y1, width, height = detections[0]['box']
			dw = round(width * border_rel)
			dh = round(height * border_rel)
			x2, y2 = x1 + width + dw, y1 + height + dh
			detected_face = img[y1:y2, x1:x2]
			detected_face = cv2.resize(detected_face, (224, 224))
			#normalize detected face in scale of -1, +1   
			# set face extraction parameters
			
			#plt.imshow(cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB))
			
			#normalize detected face in scale of -1, +1
   

			img_pixels = imageKeras.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 127.5
			img_pixels -= 1

			representation = model.predict(img_pixels)[0,:]
		except Exception as e:
			representation = None
			
		return representation

	tic = time.time()
	df['face_vector_raw'] = df['pixels'].apply(findFaceRepresentation) #vector for raw image
	toc = time.time()
	print("extracting face vectors completed in ",toc-tic," seconds...")

	tic = time.time()
	df.to_pickle("representations.pkl")
	toc = time.time()
	print("storing representations completed in ",toc-tic," seconds...")

else:
	#if you run to_pickle command once, then read pickle completed in seconds in yimageour following runs
	tic = time.time()
	df = pd.read_pickle("representations.pkl")
	toc = time.time()
	print("reading pre-processed data frame completed in ",toc-tic," seconds...")

#-----------------------------------------

print("data set: ",df.shape)






app = Flask(__name__)
CORS(app, resources={r"/prediction": {"origins": "*"}})

# cors = CORS(app, resources={r"/prediction": {"origins": "http://localhost:63701"}})

@app.route('/prediction', methods=["POST", "GET"])
def index():
	print("request received")
 
	img_path = request.form.get("path")
	if(img_path == None):
		img_path = request.json.get("path")
	# img_path = "img1.jpg"

	# print("Img_path",img_path)
	try:
		#----------------------------------------------
		#load image
		# my_image = cv2.imread(img_path)
		decoded_image = base64.b64decode(img_path)
		# Create an image object from the decoded image bytes
		nparr = np.frombuffer(decoded_image, np.uint8) #nparr = np.fromstring(decoded_image, np.uint8)
		my_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
		print(my_image.shape)
		print("img",my_image)
		detector = MTCNN()
		# set face extraction parameters
		border_rel = 0 # increase or decrease zoom on image
		detections = detector.detect_faces(my_image)
		x1, y1, width, height = detections[0]['box']
		dw = round(width * border_rel)
		dh = round(height * border_rel)
		x2, y2 = x1 + width + dw, y1 + height + dh
		face = my_image[y1:y2, x1:x2]

	
		detected_face = cv2.resize(face, (224, 224)) #resize to 224x224
		cv2.imwrite("cropped_face.jpg", detected_face)

		img_pixels = imageKeras.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		#normalize in scale of [-1, +1]
		img_pixels /= 127.5
		img_pixels -= 1

		captured_representation = model.predict(img_pixels)[0,:]			
		#----------------------------------------------

		def findCosineSimilarity(source_representation, test_representation=captured_representation):
			try:
				a = np.matmul(np.transpose(source_representation), test_representation)
				b = np.sum(np.multiply(source_representation, source_representation))
				c = np.sum(np.multiply(test_representation, test_representation))
				return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
			except:
				return 10 #assign a large value. similar faces will have small value.

		df['similarity'] = df['face_vector_raw'].apply(findCosineSimilarity)
		#look-alike celebrity
		print(df['similarity'].head())
		min_index = df[['similarity']].idxmin()[0]
		instance = df.loc[min_index]
		print(instance)

		name = instance['celebrity_name']
		similarity = instance['similarity']
		similarity = (1 - similarity)*100

		#print(name," (",similarity,"%)")

		if similarity > 50:
			full_path = instance['full_path'][0]
			celebrity_img = cv2.imread("imdb_data_set/%s" % full_path)
			celebrity_img = cv2.resize(celebrity_img, (112, 112))

			
		print("\n\nNom : ",name,"\n\n")
		imageView = Image.open("imdb_data_set/%s" % full_path)

		# Affichez l'image
		imageView.show()
		encoded_string = None
		with open("imdb_data_set/%s" % full_path, "rb") as image_file:
			image_bytes = image_file.read()
			encoded_string = base64.b64encode(image_bytes).decode("utf-8")
		de = {
			"status":200,
			"data":{
				"nom":name,
				"similarity":similarity,
				"image":encoded_string,
       	    }
		}
		response = jsonify(de)
		response = make_response(response)
		response = makeRequestHeaders(response)
		print(response)
		return response
	except Exception as e:
		print("error", e, e.__traceback__.tb_lineno)
		de = {
			"status":400,
			"data":"not working"
			}
		response = jsonify(de)
		response = make_response(response)
		response = makeRequestHeaders(response)
		print(response)
		return response



def makeRequestHeaders(response):
    try:
        myUrl = request.environ['HTTP_ORIGIN']
    except KeyError:
        print("KeyError lol")
        myUrl = '*'
    response.headers.add('Access-Control-Allow-Origin', myUrl)
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Content-type', 'application/json')
    response.headers.add('charset', 'utf8')
    return response

if __name__ == '__main__':
    app.run()