from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from LSTM_classifier import predict_category

categories_dict = {
    "귀코목질환": "ENT",
    "뇌신경정신질환": "Neuropsych",
    "소아청소년질환": "Pediatrics",
    "소화기질환": "Gastrointestinal",
    "여성질환": "Gynecology",
    "유방내분비질환": "Breast_Endocrine",
    "치과질환": "Dental",
    "호흡기질환": "Respiratory"
}



# 1. 모델 및 벡터스토어 초기화
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 벡터 임베딩 모델
client = chromadb.PersistentClient("./chroma_db_storage")

# gemma 모델 로드
tokenizer = AutoTokenizer.from_pretrained("./peft/gemma2b-it-ko-mb")  # finetuned-gemma-model 경로로 변경
bnb_config = BitsAndBytesConfig(load_in_8bit=True)#, llm_int8_enable_fp32_cpu_offload=True)


# Load model with quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    './peft/gemma2b-it-ko-mb',
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
        return response_text



# 3. 질문을 받아 RAG 수행하는 함수
def generate_answer_with_rag(question):
    # (1) LSTM으로 category 예측
    category, max_prob = predict_category(question)
    print('예측된 카테고리: ', category, max_prob)

    question = f'<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n'

    context = ""
    if category not in categories_dict.keys():
        print('연관 문서가 없습니다!')
    else:
        # (2) 예측된 category에 맞는 벡터스토어 로드
        collection_name = f"{categories_dict[category]}_chroma_db"
        collection = client.get_collection(name=collection_name)
        
        # (3) 질문을 벡터로 임베딩하고 유사한 response 검색
        question_embedding = embedding_model.encode(question)
        results = collection.query(query_embeddings=[question_embedding], n_results=5)
        
        # 검색 결과에서 response 텍스트 가져오기
        context_texts = [result["document"] for result in results["documents"][0]]
        context = " ".join(context_texts)  # 검색된 response를 하나의 문맥으로 결합
        print('검색된 document: ', context)
    
    # (4) gemma 모델로 답변 생성 (RAG 수행)
    inputs = tokenizer(question + " " + context, return_tensors="pt", truncation=True, max_length=256).to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=256, 
            do_sample=True, 
            top_k=50,  # 상위 k개의 토큰만 샘플링
            top_p=0.95  # 누적 확률 0.95까지 샘플링
            )    
        
    # 생성된 답변 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_response(response)
    return {"response": response}

# 예시 질문
while True:
    user_input = input('질문 입력: ')
    if user_input == 'quit':
        break
    #user_input = f'<bos><start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n'
    print(generate_answer_with_rag(user_input))