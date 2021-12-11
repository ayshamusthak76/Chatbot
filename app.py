# BERT
import os
import math
import datetime

import sys
sys.path


from tqdm import tqdm
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

#qna imports
import torch
# from transformers import BertForQuestionAnswering
print(tf.__version__)

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
from tensorflow.keras.models import load_model
# model = load_model('model.h5')
model = load_model('D:/Aysha/sem 7/pbl/ChatbotB/ChatbotB/colab.h5', custom_objects={"BertModelLayer": bert.model.BertModelLayer})
# print(model.summary())
import json
import random
intents = json.loads(open('D:/Aysha/sem 7/pbl/ChatbotB/ChatbotB/bert_ansrs.json').read())
words = pickle.load(open('D:/Aysha/sem 7/pbl/ChatbotB/ChatbotB/texts.pkl','rb'))
classes = pickle.load(open('D:/Aysha/sem 7/pbl/ChatbotB/ChatbotB/labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


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
    print("pred_token_ids",pred_token_ids)
    print(pred_token_ids.size)
    predictions = model.predict(pred_token_ids).argmax(axis=-1)
    print("predictions",predictions)
    return predictions

def getResponse2(ints, intents_json, msg):
    for text, label in zip(msg, ints):
        print("text:", text, "\nintent:", classes[label])
        tag= classes[label]
    intents_part = intents["intents"]
    for i in intents_part:
        print(i["tag"], tag)
        if i['tag']==tag:
            return i["responses"]
    return i["Sorry...Can u repeat"]


# linda protocol etc...
# object detection
# lora
# 6fox

def getResponse(ints, intents_json, msg):
    for text, label in zip(msg, ints):
        print("text:", text, "\nintent:", classes[label])
        return classes[label]

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            #here change
            
            ## responses_text = random.choice(i['responses'])
            responses_text = i['responses']
            # answer_question(msg, responses_text)

            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    
    ints = predict_class2(msg, model)
    print("INTS: ",ints)
    # ints = predict_class(msg, model)
    res = getResponse2(ints, intents,msg)
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()



# Qnamodel = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# def answer_question(question, answer_text):
#     '''
#     Takes a `question` string and an `answer_text` string (which contains the
#     answer), and identifies the words within the `answer_text` that are the
#     answer. Prints them out.
#     '''
#     # ======== Tokenize ========
#     # Apply the tokenizer to the input text, treating them as a text-pair.
#     input_ids = tokenizer.encode(question, answer_text)

#     # Report how long the input sequence is.
#     print('Query has {:,} tokens.\n'.format(len(input_ids)))

#     # ======== Set Segment IDs ========
#     # Search the input_ids for the first instance of the `[SEP]` token.
#     sep_index = input_ids.index(tokenizer.sep_token_id)

#     # The number of segment A tokens includes the [SEP] token istelf.
#     num_seg_a = sep_index + 1

#     # The remainder are segment B.
#     num_seg_b = len(input_ids) - num_seg_a

#     # Construct the list of 0s and 1s.
#     segment_ids = [0]*num_seg_a + [1]*num_seg_b

#     # There should be a segment_id for every input token.
#     assert len(segment_ids) == len(input_ids)

#     # ======== Evaluate ========
#     # Run our example through the model.
#     outputs = Qnamodel(torch.tensor([input_ids]), # The tokens representing our input text.
#                     token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
#                     return_dict=True) 

#     start_scores = outputs.start_logits
#     end_scores = outputs.end_logits

#     # ======== Reconstruct Answer ========
#     # Find the tokens with the highest `start` and `end` scores.
#     answer_start = torch.argmax(start_scores)
#     answer_end = torch.argmax(end_scores)

#     # Get the string versions of the input tokens.
#     tokens = tokenizer.convert_ids_to_tokens(input_ids)

#     # Start with the first token.
#     answer = tokens[answer_start]

#     # Select the remaining answer tokens and join them with whitespace.
#     for i in range(answer_start + 1, answer_end + 1):
        
#         # If it's a subword token, then recombine it with the previous token.
#         if tokens[i][0:2] == '##':
#             answer += tokens[i][2:]
        
#         # Otherwise, add a space then the token.
#         else:
#             answer += ' ' + tokens[i]

#     print('Answer: "' + answer + '"')

# import textwrap

# # Wrap text to 80 characters.
# wrapper = textwrap.TextWrapper(width=80) 

# bert_abstract = "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."

# print(wrapper.fill(bert_abstract))

# question = "What does the 'B' in BERT stand for?"

# answer_question(question, bert_abstract)



# import seaborn as sns
# from pylab import rcParams
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# from matplotlib import rc

# from sklearn.metrics import confusion_matrix, classification_report

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

# sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

# sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

# rcParams['figure.figsize'] = 12, 8

# RANDOM_SEED = 42

# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)
