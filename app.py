# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 23:46:21 2023

@author: sahilbatra
"""
from flask import Flask, request, render_template
from flask_cors import CORS
from flask import jsonify
import sys
sys.path.append('C:/Users/sahilbatra/OneDrive - Microsoft/Documents/Projects')
from WhatsAppOpenAiChatBot import func

app = Flask(__name__, template_folder='C:/Users/sahilbatra/OneDrive - Microsoft/Documents/Projects/WhatsAppOpenAIChatBot')
CORS(app) 

@app.route('/')
def home():
   #return "Hello World"
    return render_template('index.html')

@app.route('/output', methods=['POST'])
def output():
    print(request)
    return jsonify("Success")

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=5001)
    print(app.template_folder)
