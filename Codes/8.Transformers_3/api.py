from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

app = FastAPI()
print("Loading model...")

pt_model = AutoModelForTokenClassification.from_pretrained("./pt_save_pretrained_ner")
pt_tokenizer = AutoTokenizer.from_pretrained("./pt_save_pretrained_ner")
#predict

CLF = pipeline("ner", model=pt_model, tokenizer=pt_tokenizer)

print("loaded tokenizer and model")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    text: str
    res: str

def BeatutifyResult(pipeline):
    result=''
    for i in pipeline:
        result += i['word'] + ' ' + str(i['entity']) + ' \n'
        result+=' '
    return result

@app.post("/predict", response_model=Response)
def predict(request: Request):
    #inputs = pt_tokenizer(request.text)
    #outputs = pt_model(**inputs)
    
    return Response(text=request.text, res=BeatutifyResult(CLF(request.text)))