import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 데이터 읽기
data = pd.read_csv('data/sanbuldata.csv')

# 입력(X)과 출력(y) 분리
X = data[['eastsea', 'westsea_anomaly', 'eastsea_anomaly',
          'eastchina_anomaly', 'eastasia_anomaly', 'global_anomaly', 'mungyeong_temp']]
y = np.log1p(data['burned_area'])

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 전처리
full_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

X_train_prepared = full_pipeline.fit_transform(X_train)
X_valid_prepared = full_pipeline.transform(X_valid)

# 모델
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[X_train_prepared.shape[1]]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')

# 학습
history = model.fit(X_train_prepared, y_train, epochs=50, validation_data=(X_valid_prepared, y_valid))

# 🔥 loss 그래프 그리기
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.grid()
plt.show()

# 🔥 예측 테스트
test_pred = model.predict(X_valid_prepared[:5])
print("✅ 테스트 예측값:", test_pred.flatten())

# 저장
model.save('fires_model.keras')
joblib.dump(full_pipeline, 'models/full_pipeline.pkl')

print("✅ 모델과 전처리 저장 완료!")


