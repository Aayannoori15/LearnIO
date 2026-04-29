from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch 
import re 
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


app = FastAPI(title='LearnIO',description='one stop solution for all learning needs',version='1.0.0')
app.mount("/static", StaticFiles(directory="."), name="static")
model=T5ForConditionalGeneration.from_pretrained('./saved_summary_model')
tokenizer=T5Tokenizer.from_pretrained('./saved_summary_model')

import torch 
if torch.mps.is_available():
    device=torch.device('mps')
elif torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
model.to(device)

templates=Jinja2Templates(directory='.')

class dialogue_input(BaseModel):
    dialogue:str

def clean_data(text):
    text=re.sub(r'\r\n',' ',text)
    text=re.sub(r'\s+',' ',text)
    text=re.sub(r'<.*?>','',text)
    text=text.strip().lower()
    return text

def summarize_dialogue(dialogue):
        dialogue=clean_data(dialogue)
        inputs=tokenizer(dialogue,padding='max_length',max_length=512,truncation=True,return_tensors='pt').to(device)
        targets=model.generate(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],max_length=150,num_beams=4,early_stopping=True)
        summary = tokenizer.decode(targets[0], skip_special_tokens=True)
        return summary

@app.post('/summarize')
async def summarize(dialogue_entered:dialogue_input):
    summary=summarize_dialogue(dialogue_entered.dialogue)
    return {"summary":summary}

@app.get('/',response_class=HTMLResponse)
async def root(request:Request):
    return templates.TemplateResponse(request=request, name='home.html')


    
