from flask import Flask, render_template, request
import numpy as np #multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions
import cv2
from keras.models import load_model #using tensorflow libraries
import webbrowser  #The webbrowser module provides a high-level interface to allow displaying web-based documents to users.

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

haarcascade = "haarcascade_frontalface_default.xml" #taking xml file
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
print("+"*50, "loadin gmmodel")
model = load_model('model.h5') #to import some models
cascade = cv2.CascadeClassifier(haarcascade)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/choose_singer', methods = ["POST"])
def choose_singer():
	info['language'] = request.form['language']
	print(info)
	return render_template('choose_singer.html', data = info['language'])


@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():
	info['singer'] = request.form['singer']

	found = False

	cap = cv2.VideoCapture(0)
	while not(found):
		_, frm = cap.read()
		gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)

		faces = cascade.detectMultiScale(gray, 1.4, 1)

		for x,y,w,h in faces: #taking image input as face and creating the file
			found = True
			roi = gray[y:y+h, x:x+w]
			cv2.imwrite("static/face.jpg", roi)

	roi = cv2.resize(roi, (48,48))

	roi = roi/255.0   #A ROI allows us to operate on a rectangular subset of the image.
	
	roi = np.reshape(roi, (1,48,48,1))

	prediction = model.predict(roi)

	print(prediction)

	prediction = np.argmax(prediction)
	prediction = label_map[prediction]

	cap.release()

	link  = f"https://www.youtube.com/results?search_query={info['singer']}+{prediction}+{info['language']}+song"
	webbrowser.open(link)  #opening link on the web-browser

	return render_template("emotion_detect.html", data=prediction, link=link)

if __name__ == "__main__":
	app.run(debug=True)