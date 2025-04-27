# 🔹 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 🔹 1. 데이터 읽기
data = pd.read_csv('data/sanbuldata.csv')

# 🔹 2. 입력(X)과 출력(y) 분리
X = data[['eastsea', 'westsea_anomaly', 'eastsea_anomaly',
          'eastchina_anomaly', 'eastasia_anomaly', 'global_anomaly','mungyeong_temp']]
y = data['burned_area']

# 🔹 3. 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 4. 전처리 파이프라인 만들기
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# 전체 파이프라인 (여기서는 전부 수치형이니까 바로 적용)
full_pipeline = num_pipeline

# 학습 데이터 변환
X_train_prepared = full_pipeline.fit_transform(X_train)
X_valid_prepared = full_pipeline.transform(X_valid)

# 🔹 5. Keras 모델 만들기
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[X_train_prepared.shape[1]]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)  # 출력층 (회귀 문제)
])

model.compile(loss='mean_squared_error', optimizer='adam')

# 🔹 6. 모델 학습
history = model.fit(X_train_prepared, y_train, epochs=50,
                    validation_data=(X_valid_prepared, y_valid))

# 🔹 7. 모델과 파이프라인 저장
model.save('fires_model.keras')
joblib.dump(full_pipeline, 'models/full_pipeline.pkl')

print("✅ 모델 및 전처리기 저장 완료!")
