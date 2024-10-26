import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

# 데이터 준비
data = pd.read_csv('./dataset/healthcare_sampled.csv')

# 질문과 카테고리 추출
questions = data['question']
categories = data['category']

# 형태소 분석기 초기화
okt = Okt()

# 불용어 로드
stopword_file_path = "./lstm/stopword.txt"
with open(stopword_file_path, 'r', encoding='utf-8') as file:
    stopwords = set(file.read().splitlines())

# 키워드 추출 함수
def extract_keywords(text):
    nouns = okt.nouns(text)
    filtered_nouns = [word for word in nouns if word not in stopwords]
    return ' '.join(filtered_nouns)

# 질문 데이터 키워드 추출
questions = questions.apply(extract_keywords)

# 라벨 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories)
print(label_encoder.classes_)

# 텍스트 토크나이징
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)

# 패딩
X = pad_sequences(sequences, maxlen=max_len)
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# 훈련, 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# LSTM 모델 정의
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

# 하이퍼파라미터 설정
vocab_size = max_words
embed_size = 128
hidden_size = 64
output_size = len(label_encoder.classes_)

# 모델 초기화
model = LSTMClassifier(vocab_size, embed_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 모델 학습 및 로스 저장
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 결과 저장 디렉토리 생성
os.makedirs('./lstm', exist_ok=True)

# 로스 시각화 및 저장
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('./lstm/training_loss.png')
plt.close()

# 모델 저장
torch.save(model.state_dict(), './lstm/healthcare_lstm_model.pth')

# 토크나이저 저장
with open('./lstm/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# 라벨 인코더 저장
with open('./lstm/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
