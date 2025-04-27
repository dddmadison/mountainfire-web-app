# ✅ check_weights.py
# 학습한 모델의 'burned_time' 입력 가중치(weight)를 확인하는 코드

import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. 모델 로드
model = keras.models.load_model('fires_model.keras')

# 2. 첫 번째 Dense 레이어 가져오기
first_layer = model.layers[0]
weights, biases = first_layer.get_weights()

# 3. Weights 모양 출력 (특성 수, 뉴런 수)
print(f"✅ Weights shape: {weights.shape}")

# 4. 'burned_time' feature의 index 찾기
# 주의: 입력 순서에 따라 다르지만, 일반적으로
# ['eastsea', 'westsea_anomaly', 'eastsea_anomaly', 'eastchina_anomaly', 'eastasia_anomaly', 'global_anomaly', 'burned_time', 'mungyeong_temp']
# 이렇게 구성했으면, burned_time은 7번째 (index=6)
burned_time_idx = 6  # 0부터 시작

# 5. burned_time 관련 weight들만 추출
burned_time_weights = weights[burned_time_idx]

# 6. 결과 출력
print("\n✅ burned_time 연결된 weights:")
print(burned_time_weights)

# 7. 평균값으로 중요도 판단
print("\n✅ burned_time weight 평균:", np.mean(burned_time_weights))

# 추가로 절댓값 평균도 같이 보기 (영향력 강도)
print("✅ burned_time weight 절댓값 평균:", np.mean(np.abs(burned_time_weights)))

# (선택) 만약 weight들이 전부 0에 가깝다면 → burned_time은 무시됐다는 뜻
# (선택) 만약 꽤 큰 weight를 가지면 → 모델이 burned_time을 고려했다는 뜻
