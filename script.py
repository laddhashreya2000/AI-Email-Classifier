#importing libraries
import os
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from flask import send_file
import numpy as np
import pandas as pd
import sys
import re
import  string
import math
import extract_msg
import flask
import pickle
import glob 
from zipfile import ZipFile
from flask import Flask, render_template, request,flash
from flask import send_file
from flask import jsonify
from werkzeug.utils import secure_filename

# final results
finaldata = pd.DataFrame()

UPLOAD_FOLDER = 'dataset'

#creating instance of the class
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')
    #return "Hello World"

#prediction function
def ValuePredictor(testdata):
    # to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("models/pickle_model_svc.pkl","rb"))
    result = loaded_model.predict(testdata['content'])
    print(result)
    df = testdata
    dic={}
    dic[0]="Retirements"
    dic[1]="MDU"
    dic[2]="Transfers"
    df['label']=[dic[wo] for wo in result]
    finaldata['label']=df['label']

    finaldata.to_csv('ans_elim.csv')
    print('hello')
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        param1 = request.form['name']
        param2 = request.form['email']
        param3 = request.form['phone']
        if 'file' not in request.files:
            return 'Get Lost'
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        files , file_extension = os.path.splitext(app.config['UPLOAD_FOLDER'] +"/"+ filename)
        print(file_extension)
        print(file.filename)
        if file_extension == '.zip':                
            with ZipFile(app.config['UPLOAD_FOLDER']+"/"+filename, 'r') as zip_ref:   #files is directory for saving mydata.zip
                zip_ref.extractall("extract")      # directory for extract 
        print(file.filename.replace(file_extension,''))              
        data =  preprocessing("extract"+"/"+file.filename.replace(file_extension,''))
        # finaldata=pd.DataFrame()
        # names=[]
        # for file in files
        print(data)
        result = ValuePredictor(data)
        print(result)
            
        return render_template("result.html",result=result)
        # return jsonify(first=param1,second=param2,third=param3)




@app.route("/getPlotCSV")
def getPlotCSV():
    # with open("outputs/Adjacency.csv") as fp:
    #     csv = fp.read()
    # csv = '1,2,3\n4,5,6\n'
    print("heeee")
    return send_file('ans_elim.csv',
                     mimetype='text/csv',
                     attachment_filename='clasify_elim.csv',
                     as_attachment=True)

@app.route('/showcsv')
def showcsv():
    filename = 'ans_elim.csv'
    data = pd.read_csv(filename, header=0)
    listval = list(data.values)
    return render_template('values.html', stocklist=listval)

def preprocessing(pathtofolder):

    # label = []
    content = []
    data = pd.DataFrame()
    path = pathtofolder+'/*.msg'
    print(path)
    files=glob.glob(path) 
    print(len(files))
    names = []
    for file in files:     
        f=open(file, 'r')  
        # print(f.name.substr(31))
        names.append(f.name[33:])
        # print(file)
        msg = extract_msg.Message(file)
        msg_subj = msg.subject
        msg_body = msg.body
        # label.append("mdu")
        content.append(msg_subj + msg_body)

    finaldata['name']=names    

    data["content"] = content
        #removing null spaces
    print(data["content"].isnull().sum())
    # print(data["label"].isnull().sum())

    data = data.dropna(axis=0, subset=['content']).reset_index(drop=True)
    # data = data.dropna(axis=0, subset=['title']).reset_index(drop=True)
    #website, URLs removal
    pattern2 = r"(https?:\/\/) (\s)*(www\.)?(\s)* ((\w|\s)+\.)*([\w\-\s]+\/)*([\w-]+) ((\?) ?[\w\s]*=\s+[\w\%&]*)*"
    pattern1 = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"

    data['content'] = data['content'].str.replace(pattern1, '')

    data['content'] = data['content'].str.replace(pattern2, '')
    #removing html tags
    html = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    data['content'] = data['content'].str.replace(html, ' ')
    email = re.compile('\S*@\S*\s')
    data['content'] = data['content'].str.replace(email, '')
    #cleaning tags
    tags = re.compile('@\S*\s')
    data['content'] = data['content'].str.replace(r'@\S*\s', '')
    #removing punctuations
    data['content'] = data['content'].str.replace(r'[^\w\s]', ' ')
    #removing non ascii characters


    data['content'] = data['content'].astype("string", copy = False)

    data['content'] = data['content'].apply(lambda x: x.encode('ascii','ignore').decode())
    # data['title'] = data['title'].apply(lambda x: x.encode('ascii','ignore').decode())

        # df["label"] = label    
    #removing numerical data
    data['content'] = data['content'].str.replace(r'\d+', '')
    

    #removing stopwords
    new_stp = ["FW:", "fw", "RE:", "re", "URGENT:", "urgent", "abc", "xyz", "from", "to", "subject", "test", "i", "sent", "to"]
    stp = stopwords.words('english')
    stp.extend(new_stp)


    pattern = re.compile(r'\b(' + r'|'.join(stp) + r')\b\s*')
    data['content'] = data['content'].str.replace(pattern, '') 
    #Converting Dataset into lower-case
    data['content'] = data['content'].str.lower()
    #removing extra space
    data['content'] = data['content'].str.replace(r'\s+', ' ')   

    key_words={}

    for i in data['content']:
      arr = i.split(' ')
      for word in arr:
        if word in key_words.keys():
          key_words[word]+=1
        else:
          key_words[word]=1


    for i in key_words.keys():
      if key_words[i]<3:
        stp.append(i)



    pattern = re.compile(r'\b(' + r'|'.join(stp) + r')\b\s*')
    data['content'] = data['content'].str.replace(pattern, '')
    nlp = spacy.load("en_core_web_sm")

    # keep_pos = ['PROPN', 'NOUN']
    data['t1'] = ''
    i=0
    for text in data['content']:
        data['t1'][i] = ' '.join([str(tok.lemma_) for tok in nlp(str(text))])
        i+=1
    # df = df.sample(frac=1).reset_index(drop=True)
    # df.to_csv('train_3.csv')
    # data = pd.read_csv('train_3.csv')
    return data
    # df.head()


if __name__ == "__main__":
	app.run(debug=True)