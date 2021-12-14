# BERT
import os
import math
import datetime
import re
import io
import sys
sys.path
import zlib
from werkzeug.utils import secure_filename
from flask import Response
from cs50 import SQL
from flask import Flask, flash, jsonify, redirect, render_template, request, session ,url_for
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import face_recognition
from PIL import Image
from base64 import b64encode, b64decode
from tqdm import tqdm
import pandas as pd
import numpy as np
from helpers import apology, login_required
import tensorflow 
from tensorflow import keras
import cv2
import time
from threading import Thread
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

#qna imports
import torch
# from transformers import BertForQuestionAnswering

global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
thres=0
face=0
switch=1
rec=0

try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('D:/Aysha/sem 7/pbl/ChatbotB - Copy (2)/ChatbotB/saved_model/deploy.prototxt.txt', 'D:/Aysha/sem 7/pbl/ChatbotB - Copy (2)/ChatbotB/saved_model/res10_300x300_ssd_iter_140000.caffemodel')

bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir=os.path.join("D:/Aysha/sem 7/pbl/ChatbotB/",bert_model_name)
# bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

# AI CHATBOT
import nltk
# nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
# import numpy as np
import bert
from bert import BertModelLayer
from transformers import BertForQuestionAnswering
from tensorflow.keras.models import load_model
model = load_model('D:/Aysha/sem 7/pbl/ChatbotB/ChatbotB/colab.h5', custom_objects={"BertModelLayer": bert.model.BertModelLayer})
import json
import random

print(tensorflow.__version__)

#BERT QNA STUFF


Qnamodel = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer

tokenizer_Qna = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer_Qna.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer_Qna.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example through the model.
    outputs = Qnamodel(torch.tensor([input_ids]), # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                    return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer_Qna.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')
    return answer

import textwrap

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80) 

intents = json.loads(open('D:/Aysha/sem 7/pbl/ChatbotB/ChatbotB/bert_ansrs.json').read())

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
train = pd.read_csv("D:/Aysha/sem 7/pbl/ChatbotB/ChatbotB/train.csv")
classes = train.intent.unique().tolist()
# classes = bert_ansrs['intents']
print(classes)
def predict_class2(msg,model):
    print((msg))
    sentence =[msg]
    pred_tokens = map(tokenizer.tokenize, sentence)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

    pred_token_ids = map(lambda tids: tids +[0]*(12-len(tids)),pred_token_ids)
    pred_token_ids = np.array(list(pred_token_ids))
    # print("pred_token_ids",pred_token_ids)
    # print(pred_token_ids.size)
    predictions = model.predict(pred_token_ids).argmax(axis=-1)
    # print("predictions: ", classes[predictions])
    # score = tensorflow.nn.softmax(predictions)
    # print("accuracy", classes[np.argmax(score)])
    # predictions2 = model.predict(pred_token_ids)[-1]
    # print(predictions2)
    # print("predictions: ")
    # score2 = tensorflow.nn.softmax(predictions2)
    # print("accuracy", classes[np.argmax(score2)],np.argmax(score2))
    # print("predictions",predictions(score2))
    # print("Accuracy")
    return predictions

def getResponse2(ints, intents_json, msg):
    # if accuracy<0.6:
    #     return "Sorry! Can you repeat"
    for text, label in zip(msg, ints):
        print("text:", text, "\nintent:", classes[label])
        tag= classes[label]
    intents_part = intents["intents"]
    for i in intents_part:
        print(i["tag"], tag)
        if i['tag']==tag:
            #add QnA part
            ansr = answer_question("what is "+ msg, i["responses"])
            return ansr
            # return i["responses"]
    

def chatbot_response(msg):    
    ints = predict_class2(msg, model)
    res = getResponse2(ints, intents,msg)
    return res


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
 



from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     return chatbot_response(userText)

#starto
# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Custom filter


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///D:/Aysha/sem 7/pbl/ChatbotB/ChatbotB/users.db")

@app.route("/")
@login_required
def home():
    return redirect("/home")

@app.route("/home")
@login_required
def index():
    return render_template("index copy.html")

@app.route("/get")
def get_bot_response():
    print("AM in GET")
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")

        # Ensure username was submitted
        if not input_username:
            return render_template("login.html",messager = 1)

        # Ensure password was submitted
        elif not input_password:
             return render_template("login.html",messager = 2)

        # Query database for username
        username = db.execute("SELECT * FROM users WHERE username = :username",
                              username=input_username)

        # Ensure username exists and password is correct
        if len(username) != 1 or not check_password_hash(username[0]["hash"], input_password):
            return render_template("login.html",messager = 3)

        # Remember which user has logged in
        session["user_id"] = username[0]["id"]



        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")



@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")
        input_confirmation = request.form.get("confirmation")

        # Ensure username was submitted
        if not input_username:
            return render_template("register.html",messager = 1)

        # Ensure password was submitted
        elif not input_password:
            return render_template("register.html",messager = 2)

        # Ensure passwsord confirmation was submitted
        elif not input_confirmation:
            return render_template("register.html",messager = 4)

        elif not input_password == input_confirmation:
            return render_template("register.html",messager = 3)

        # Query database for username
        username = db.execute("SELECT username FROM users WHERE username = :username",
                              username=input_username)

        # Ensure username is not already taken
        if len(username) == 1:
            return render_template("register.html",messager = 5)

        # Query database to insert new user
        else:
            new_user = db.execute("INSERT INTO users (username, hash) VALUES (:username, :password)",
                                  username=input_username,
                                  password=generate_password_hash(input_password, method="pbkdf2:sha256", salt_length=8),)

            if new_user:
                # Keep newly registered user logged in
                session["user_id"] = new_user

            # Flash info for the user
            flash(f"Registered as {input_username}")

            # Redirect user to homepage
            return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")


@app.route("/facereg", methods=["GET", "POST"])
def facereg():
    session.clear()
    if request.method == "POST":


        encoded_image = (request.form.get("pic")+"==").encode('utf-8')
        username = request.form.get("name")
        name = db.execute("SELECT * FROM users WHERE username = :username",
                        username=username)
              
        if len(name) != 1:
            return render_template("camera.html",message = 1)

        id_ = name[0]['id']    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        try:
            image_of_bill = face_recognition.load_image_file(
            './static/face/'+str(id_)+'.jpg')
        except:
            return render_template("camera.html",message = 5)

        bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

        unknown_image = face_recognition.load_image_file(
        './static/face/'+str(id_)+'.jpg')
        try:
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        except:
            return render_template("camera.html",message = 2)


#  o    mpare faces
        results = face_recognition.compare_faces(
        [bill_face_encoding], unknown_face_encoding)

        if results[0]:
            username = db.execute("SELECT * FROM users WHERE username = :username",
                              username="swa")
            session["user_id"] = username[0]["id"]
            return redirect("/")
        else:
            return render_template("camera.html",message=3)


    else:
        return render_template("camera.html")



@app.route("/facesetup", methods=["GET", "POST"])
def facesetup():
    if request.method == "POST":


        encoded_image = (request.form.get("pic")+"==").encode('utf-8')


        id_=db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]
        # id_ = db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        image_of_bill = face_recognition.load_image_file(
        './static/face/'+str(id_)+'.jpg')    
        try:
            bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]
        except:    
            return render_template("face.html",message = 1)
        return redirect("/home")

    else:
        return render_template("face.html")

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(thres):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask, frame = cv2.threshold(gray, 120,255,cv2.THRESH_BINARY)
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
            
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route('/filters')
def filters():
    return render_template('filters.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('thres') == 'Threshold':
            global thres
            thres=not thres
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('filters.html')
    return render_template('filters.html')

# from filters import verify_alpha_channel
# @app.route('/serpia')
# def apply_sepia(frame, intensity=0.5):
#     frame = verify_alpha_channel(frame)
#     frame_h, frame_w, frame_c = frame.shape
#     sepia_bgra = (20, 66, 112, 1)
#     overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
#     cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
#     return frame



def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return render_template("error.html",e = e)

# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

#end
if __name__ == "__main__":
    app.run()
