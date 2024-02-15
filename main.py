import os
from app import app
import numpy as np
import logging
import cv2
import glob
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response, jsonify
from werkzeug.utils import secure_filename
import time
import subprocess
from multiprocessing import Process

import json
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


@app.route('/', methods=['POST'])
def upload_video():
    global executed
    executed = False

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_video filename: ' + filename)
        global message_to_display
        # message_to_display = "Video successfully uploaded..."
        # flash('Video successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    
     


# cap = cv2.VideoCapture('./static/Apprendre_langlais_avec_des_videos___British_Council_France_-_Google_Chrome_2023-01-16_14-05-06.mp4')
def gen():
    
    
    videos_path = './static/uploads'

    # Récupérer les chemins des vidéos avec l'extension mp4
    video_paths = glob.glob(f"{videos_path}/*.mp4")
    video_paths.sort(key=os.path.getmtime, reverse=True)
    print(video_paths)
    cap = cv2.VideoCapture(video_paths[0])
    while(True):
        for i in range(21):
            
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 10, 200, apertureSize=3)
            vis = img.copy()
            vis = np.trunc(vis / 2)
            vis[edge != 0] = (0, 255, 0)
            merge = np.concatenate((img, vis), axis=1)
            cv2.imwrite('./examples/demo/' + str(i+1) + '.png', img)
            if i == 5:
                print("image5")
                cv2.imwrite('./static/images/' + str(i+1) + '.png', img)
                
            elif i == 20:
                 print("images20")
                 cv2.imwrite('./static/images/' + str(i+1) + '.png', img)
        break   
        # frame = open('out.png', 'rb').read()
        
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def preprocess():
     # Charger le fichier JSON
    with open('./static/poses/alphapose-results.json') as file:
        data = json.load(file)

    keypoints_list = []
    for item in data:

        keypoints = item['keypoints']
        
        keypoints_list.append(keypoints)

    keypoints_array = np.array(keypoints_list)
  
    return keypoints_array
    
def predict(keypoints_array):
    n_input = 51
    n_steps = 21
    n_hidden = 20

    def LSTM_RNN(_X, _weights, _biases):
        # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        _X = tf.reshape(_X, [-1, n_input])   
        # Rectifies Linear Unit activation function used
        _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, n_steps, 0) 

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        outputs, states = tf.compat.v1.nn.static_rnn(lstm_cells, _X, dtype=tf.float32)

        # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
        lstm_last_output = outputs[-1]
        
        # Linear activation
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, 21, 51])
    y = tf.placeholder(tf.float32, [None, 3])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random.normal([51, 20])), # Hidden layer weights
        'out': tf.Variable(tf.random.normal([20, 3], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random.normal([20])),
        'out': tf.Variable(tf.random.normal([3]))
    }

    pred = LSTM_RNN(x, weights, biases)

    model = "./static/DanceModel/danceClassify.ckpt-411"
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)
    # saver.restore(sess, model)

    predictions = sess.run(pred, feed_dict={x: np.expand_dims(keypoints_array, axis=0)})
    print(predictions)
    LABELS =["afrobeat", "classical", "hiphop"]
    max_index = np.argmax(predictions)
    # get the index of the max probability
    label = LABELS[max_index]
    return label

executed = False

@app.route('/check-images')
def check_images():

  if os.listdir('./static/images'):
    return '{"hasImages": true}' 
  else:
    return '{"hasImages": false}'  
  

@app.route('/check-poses')
def check_poses():
  poses_dir = './static/poses/vis'

  if os.path.exists(poses_dir):
    return '{"hasPoses": true}' 
  else:
    return '{"hasPoses": false}'


@app.route('/predict')
def posedetect():
    global executed
    executed = False
    
    if not executed:
        
        command = 'python demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo --outdir ./static/poses --save_img --eval --gpu 0,1,2,3'
        subprocess.run(command, shell=True)
        
        executed = True
    
    time.sleep(20)
    print("enter to label predict")
    label = predict(preprocess())
    print("label", label)
    
    
    return {'label': label}

     
@app.route('/display/<filename>')
def display_video(filename):
    print(filename)
    # # Répertoire des vidéos
    # video_folder = os.path.join(app.config['UPLOAD_FOLDER'])
    # print(filename)
    # # Supprimer les anciennes vidéos
    # for file in os.listdir(video_folder):
    #     file_path = os.path.join(video_folder, file)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)
    time.sleep(60)
    return redirect(url_for('static', filename='uploads/'  + filename), code=301)


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('upload.html')


#############Predict#################"

if __name__ == '__main__':
    app.run(debug=True)