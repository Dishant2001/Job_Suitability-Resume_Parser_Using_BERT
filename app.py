from werkzeug.utils import secure_filename
from flask import Flask,request
from flask_cors import CORS
#from pyresparser import ResumeParser
import json
import os
import PyPDF2 as pdfReader
from string import punctuation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import *

model = SentenceTransformer('bert-base-nli-mean-tokens',device='cpu')

app = Flask(__name__)

CORS(app,resources={r"/*": {"origins": "*"}})

def process(txt):
    txt = txt.lower()
    for i in txt:
        if i in punctuation:
            txt = txt.replace(i,' ')
        elif i == '●':
            txt = txt.replace(i,' ')
        elif i == '–':
            txt = txt.replace(i,' ')
    txt = ' '.join(txt.split())
    return txt


@app.route('/home',methods = ['POST'])

def home():
    if request.method=='POST':
        try:
            file = request.files['resume']
            jd = process(request.form['jd'])
            if ' ' in file.filename:
                file.filename = file.filename.replace(' ','_')
            file.save(secure_filename(file.filename))
            #ner2 = ResumeParser(file.filename).get_extracted_data()
            pdfFileObj = open(file.filename,'rb')
            reader = pdfReader.PdfFileReader(pdfFileObj)
            raw_text = reader.getPage(0).extractText()
            txt = process(raw_text)
            email = extractEmail(raw_text)
            url = urls(raw_text)
            pdfFileObj.close()
            os.remove(file.filename)
            sentence_embeddings = model.encode([jd,txt])
            simi = cosine_similarity([sentence_embeddings[0]],[sentence_embeddings[1]])[0][0]
            text_data,prediction = prepareData(raw_text)
            ner = token2tags(text_data,prediction)
            ner['EMAIL'] = email
            ner['URL'] = url
            data = {'similarity':simi*100,'ner':ner}
            return json.dumps(data)
        except Exception as e:
            return f"h1>{e}</h1>"
if __name__=="__main__":
    app.run(host='0.0.0.0')


