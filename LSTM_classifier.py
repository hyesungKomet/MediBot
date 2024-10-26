import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import torch.nn as nn

# 형태소 분석기 및 기타 설정
okt = Okt()
max_len = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM 모델 정의 (저장된 모델과 동일하게 정의)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 마지막 시퀀스의 출력값만 사용
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 모델, 토크나이저, 라벨 인코더 불러오기
def load_resources():
    # 모델 초기화 및 가중치 로드
    vocab_size = 10000  # max_words와 동일
    embed_size = 128
    hidden_size = 64
    output_size = 8  # 예시: 카테고리 수
    model = LSTMClassifier(vocab_size, embed_size, hidden_size, output_size)
    model.load_state_dict(torch.load('./lstm/healthcare_lstm_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # 토크나이저 로드
    with open('./lstm/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # 라벨 인코더 로드
    with open('./lstm/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder

# 키워드 추출 함수
def extract_keywords(text, stopwords):
    nouns = okt.nouns(text)
    filtered_nouns = [word for word in nouns if word not in stopwords]
    return ' '.join(filtered_nouns)

# 예측 함수
def predict_category(user_input, threshold=0.5):
    # 모델, 토크나이저, 라벨 인코더 로드
    model, tokenizer, label_encoder = load_resources()

    # 불용어 로드
    stopword_file_path = "./lstm/stopword.txt"
    with open(stopword_file_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())

    # 입력 텍스트 전처리
    keywords = extract_keywords(user_input, stopwords)
    sequences = tokenizer.texts_to_sequences([keywords])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    # Tensor로 변환하여 모델에 입력
    input_tensor = torch.tensor(padded_sequences, dtype=torch.long).to(device)

    # 예측 수행
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)  # 예측 확률 계산
        max_prob, predicted_index = torch.max(probabilities, dim=1)
        max_prob = max_prob.item()  # 최대 확률 값 추출

    # threshold 기준으로 예측
    if max_prob >= threshold:
        # 인코딩된 카테고리를 실제 라벨로 변환
        predicted_category = label_encoder.inverse_transform([predicted_index.item()])[0]
    else:
        predicted_category = "not_sure"  # threshold 미달 시 불확실로 표시

    return predicted_category, max_prob

# 예시 사용

# 예시 사용
# user_input = input("질문을 입력하세요: ")
# predicted_category = predict_category(user_input)
# print(f"예측된 카테고리: {predicted_category}")
