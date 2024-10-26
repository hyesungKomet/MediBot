from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import re

# Define the API app
# app = FastAPI()

# Load the tokenizer and model at startup
tokenizer = AutoTokenizer.from_pretrained('./peft2/gemma2b-it-ko-mb')
# Define the 8-bit quantization configuration
bnb_config = BitsAndBytesConfig(load_in_8bit=True)#, llm_int8_enable_fp32_cpu_offload=True)

# Load model with quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    './peft2/gemma2b-it-ko-mb',
    device_map="auto",
    quantization_config=bnb_config
)

model.eval()



def clean_response(response_text):
    # Remove all user turns, keeping only the first model or doctor response
    # Find the first model or doctor turn with <start_of_turn> and <end_of_turn>
    match = re.search(r"<start_of_turn>(model|doctor)\n(.*?)<end_of_turn>", response_text, re.DOTALL)
    
    # If found, return the response; otherwise, return an empty string
    if match:
        return match.group(2).strip()  # Extract and return the content inside the first model/doctor turn
    else:
        return "No valid response found."

def generate(user_input):
    # Tokenize the input question
    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=256, 
            do_sample=True, 
            top_k=50,  # 상위 k개의 토큰만 샘플링
            top_p=0.95  # 누적 확률 0.95까지 샘플링
            )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_response(response)
    return {"response": response}

while True:
    user_input = input('질문 입력: ')
    if user_input == 'quit':
        break
    user_input = f'<bos><start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n'
    print(generate(user_input))