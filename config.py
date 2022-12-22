import numpy as np
import pandas as pd
import spacy
import torch
import re
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
nlp = spacy.load('en_core_web_lg')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
device = torch.device('cpu')

def get_test_data(text):
    sentences = []
    
    doc = nlp(text) 
    doc = list(doc)
    sentences.append(doc)
    
    return sentences


def get_tokenized_test_data(sentences):

    tokenized_texts = []

    for word_list in sentences:
        temp_token = ['[CLS]']
    
        for word in word_list:
            token_list = tokenizer.tokenize(word.text)
            for m, token in enumerate(token_list):
                temp_token.append(token) 
        temp_token.append('[SEP]')
    
        tokenized_texts.append(temp_token)
    
    return tokenized_texts

def prepareData(resume):
    prefixes = ['\\n', ] + nlp.Defaults.prefixes
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefix_regex.search
    sentences = get_test_data(resume)

    idx2tag = {0: 'L-COMPANY',
    1: 'L-CLG',
    2: 'U-LOC',
    3: 'B-CLG',
    4: 'L-YOE',
    5: 'B-DESIG',
    6: 'B-COMPANY',
    7: 'B-LOC',
    8: 'I-COMPANY',
    9: 'X',
    10: 'L-DESIG',
    11: 'U-EMAIL',
    12: 'U-COMPANY',
    13: 'I-NAME',
    14: 'B-SKILLS',
    15: 'I-LOC',
    16: 'B-EMAIL',
    17: 'U-CLG',
    18: 'B-DEG',
    19: 'B-GRADYEAR',
    20: 'L-GRADYEAR',
    21: 'B-YOE',
    22: '[SEP]',
    23: 'B-NAME',
    24: 'I-EMAIL',
    25: 'I-GRADYEAR',
    26: 'L-LOC',
    27: 'U-YOE',
    28: 'I-DEG',
    29: 'I-YOE',
    30: 'L-EMAIL',
    31: 'I-DESIG',
    32: 'U-GRADYEAR',
    33: 'L-DEG',
    34: '[CLS]',
    35: 'I-CLG',
    36: 'O',
    37: 'I-SKILLS',
    38: 'U-SKILLS',
    39: 'L-NAME',
    40: 'L-SKILLS',
    41: 'U-DEG',
    42: 'U-DESIG'}

    tokenized_text = get_tokenized_test_data(sentences)
    MAX_LEN = 512
    bs = 4
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_text],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

    test_inputs = torch.tensor(input_ids)
    test_masks = torch.tensor(attention_masks)
    test_data = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)

    model = torch.load('bert_resume_ner.pth',map_location = torch.device('cpu'))
    model.eval()

    y_pred = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask = batch

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=None, attention_mask=input_mask,)

        logits = logits.detach().cpu().numpy()
        logits = [list(p) for p in np.argmax(logits, axis=2)]
        
        input_mask = input_mask.to('cpu').numpy()
      
        for i,mask in enumerate(input_mask):
            temp_2 = [] # Predict one
          
            for j, m in enumerate(mask):
                # Mark=0, meaning its a pad word, dont compare
                if m:
                    temp_2.append(idx2tag[logits[i][j]])
                else:
                    break

            y_pred.append(temp_2)
      
    return tokenized_text,y_pred

def token2tags(sentences,predictions):
    entity_dict = {
        'NAME':[], 
        'CLG':[],
        'DEG':[],
        'GRADYEAR':[],
        'YOE':[],
        'COMPANY':[],
        'DESIG':[],
        'SKILLS':[],
        'LOC':[],
        'EMAIL':[]
    }

    for i in range(1,len(predictions[0])):
        if predictions[0][i]=='X' and predictions[0][i-1]!='X' and predictions[0][i-1]!='CLS':
            predictions[0][i] = predictions[0][i-1]

    for i in entity_dict.keys():
        for idx,j in enumerate(predictions[0]):
            if i in j:
                entity_dict[i].append(idx)
    result_dict = {}
    for i in entity_dict.keys():
        result_dict[i] = ''
        for j in entity_dict[i]:
            result_dict[i]+=sentences[0][j]+' '
        result_dict[i] = result_dict[i].replace(' ##','')
    return result_dict

def extractEmail(email):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", email)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None

def urls(string):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]
