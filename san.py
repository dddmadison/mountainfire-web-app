import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pandas as pd
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib
import platform

# 한글 깨짐 방지 설정
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"   # Windows는 맑은 고딕
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"      # Mac은 애플고딕
else:
    plt.rcParams["font.family"] = "NanumGothic"      # 리눅스나 기타 환경은 나눔고딕 추천

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨지는 것 방지


fires_raw = pd.read_csv("./data/sanbuldata.csv", encoding="utf-8")
fires = fires_raw.copy()

# 🔥 month, day 컬럼 생성 (여기가 빠졌었어)
fires["산불발생일시"] = pd.to_datetime(fires["산불발생일시"], errors="coerce")
fires["month"] = fires["산불발생일시"].dt.month
fires["day"] = fires["산불발생일시"].dt.day_name()


print("\n2019210147 우상용\n")



# 1-1 불러오기
# # 데이터 불러오기
# try:
#     fires = pd.read_csv("./data/sanbuldata.csv", encoding="utf-8")
#     print("데이터 불러오기 성공")
# except FileNotFoundError:
#     print("파일을 찾을 수 없습니다.")
# except Exception as e:
#     print(f"다른 에러 발생: {e}")

# # 'burned_area' 컬럼 변환
# if "burned_area" in fires.columns:
#     fires["burned_area"] = np.log1p(fires["burned_area"])
#     print("burned_area 변환 완료 (log1p 적용)")
# else:
#     print("'burned_area' 컬럼이 없습니다.")

# 1-2 기본 정보 확인
# 데이터 로드
# fires = pd.read_csv("./data/sanbuldata.csv", encoding="utf-8")

# # burned_area log1p 변환
# fires["burned_area"] = np.log1p(fires["burned_area"])


# # ===== 데이터 요약 출력 =====
# print("\nThis is head")
# print(fires.head())
# print("\nThis is info")
# print(fires.info())
# print("\nThis is describe")
# print(fires.describe())
# print("\nMonth 산불 발생 건수")
# print(fires["month"].value_counts().sort_index())
# print("\nDay 산불 발생 건수")
# print(fires["day"].value_counts())


# 1-3
# # ===== 분석할 컬럼 =====
# numeric_cols = ["burned_area", "eastsea", "eastsea_anomaly", "문경평균기온(℃)", "산불진행시간(분)"]

# # ====== subplot 시작 ======
# fig, axes = plt.subplots(2, 3, figsize=(20, 12))
# axes = axes.flatten()

# # 1~5번: 각 수치형 변수 히스토그램
# for idx, col in enumerate(numeric_cols):
#     sns.histplot(fires[col], kde=True, bins=30, ax=axes[idx])
#     axes[idx].set_title(f"{col} 분포")
#     axes[idx].set_xlabel(col)
#     axes[idx].set_ylabel("빈도수")
#     axes[idx].grid(True)

# # 6번: 동해수온 vs 피해면적 산점도
# sns.scatterplot(x="eastsea", y="burned_area", data=fires, ax=axes[5])
# axes[5].set_title("eastsea vs burned_area")
# axes[5].set_xlabel("eastsea")
# axes[5].set_ylabel("burned_area")
# axes[5].grid(True)

# # 레이아웃 정리
# plt.tight_layout()
# plt.show()

#1-4

# # ===== 변환하기 (log1p) =====
# fires["burned_area"] = np.log1p(fires["burned_area"])

# # ===== 변환 전/후 히스토그램 비교 =====
# plt.figure(figsize=(12, 5))

# # 1. 변환 전
# plt.subplot(1, 2, 1)
# plt.hist(fires_raw["burned_area"], bins=30, color="skyblue", edgecolor="black")
# plt.title("로그 변환 전 (burned_area)")
# plt.xlabel("burned_area")
# plt.ylabel("빈도수")

# # 2. 변환 후
# plt.subplot(1, 2, 2)
# plt.hist(fires["burned_area"], bins=30, color="salmon", edgecolor="black")
# plt.title("로그 변환 후 (log1p(burned_area))")
# plt.xlabel("log(1 + burned_area)")
# plt.ylabel("빈도수")

# plt.tight_layout()
# plt.show()


#1-5

# train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

# # 🔹 Test set 일부 확인
# print("\n[Test set head]")
# print(test_set.head())

# # ===============================
# # 2. fires["month"] 히스토그램 그리기
# # ===============================
# fires["month"].hist(bins=12, figsize=(8, 6))
# plt.title("Month Distribution (전체 데이터)")
# plt.xlabel("Month")
# plt.ylabel("Count")
# plt.grid(True)
# plt.show()

# # ===============================
# # 3. StratifiedShuffleSplit으로 month 기준 stratified split
# # ===============================
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# for train_index, test_index in split.split(fires, fires["month"]):
#     strat_train_set = fires.loc[train_index]
#     strat_test_set = fires.loc[test_index]

# # ===============================
# # 4. month 분포 비율 출력
# # ===============================

# print("\nMonth category proportion (strat_test_set):\n",
#       strat_test_set["month"].value_counts(normalize=True).sort_index())

# print("\nOverall month category proportion (전체 fires):\n",
#       fires["month"].value_counts(normalize=True).sort_index())


#1-6

# # 분석할 5개 특성 선정
# selected_features = ["동중국해 해수면 온도 이상", "동아시아 해역 해수면 온도 이상", "eastsea", "burned_area"]

# # scatter_matrix 그리기
# scatter_matrix(fires[selected_features], figsize=(12, 8), alpha=0.7, diagonal="hist")

# # 제목 설정
# plt.suptitle("Scatter Matrix of Selected Features", fontsize=16)
# plt.tight_layout()
# plt.show()




#1-7

# fires.plot(kind="scatter",
#            x="문경최고기온(℃)",    # X축: 문경 최고 기온
#            y="burned_area",   # Y축: 피해 면적
#            alpha=0.4,
#            s=fires["산불진행시간(분)"],    # 원 크기: 산불 진행 시간
#            label="산불진행시간(분)",
#            c=fires["burned_area"],    # 원 색깔: burned_area
#            cmap=plt.get_cmap("jet"),
#            colorbar=True,
#            figsize=(10, 6))

# plt.title("최고기온(℃) vs 피해면적(ha)\n(산불진행시간에 따른 원 크기)")
# plt.xlabel("문경최고기온(℃)")
# plt.ylabel("burned_area")
# plt.grid(True)
# plt.legend()
# plt.show()

#1-8


# split 하기 전에 fires에 month가 2개 이상 있는 것만 필터링해야 해
month_counts = fires["month"].value_counts()
valid_months = month_counts[month_counts >= 2].index
fires = fires[fires["month"].isin(valid_months)]

fires = fires.reset_index(drop=True)  # 인덱스 초기화!! 중요!!

# 여기서 stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]  # 🔥 이걸 만들어야 해
    strat_test_set = fires.loc[test_index]


# 1. 레이블 분리
fires = strat_train_set.drop(["burned_area"], axis=1)  # drop labels for training set
fires_labels = strat_train_set["burned_area"].copy()

# 2. 수치형 특성 분리
fires_num = fires.drop(["month", "day"], axis=1)

# 3. 범주형 특성 지정
cat_attrs = ["month", "day"]

# 4. OneHotEncoder 적용
encoder = OneHotEncoder(sparse_output=False)
cat_encoded = encoder.fit_transform(fires[cat_attrs])

# 5. 변환 결과를 데이터프레임으로 변환
encoded_cols = encoder.get_feature_names_out(cat_attrs)
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoded_cols, index=fires.index)

# 6. 출력
print("\nOneHot Encoding된 컬럼 목록:")
print(list(encoded_cols))

print("\nOneHot Encoding 결과 샘플:")
print(cat_encoded_df.head())


# # 1-9

print("\n\n########################################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")

# 🔹 수치형, 범주형 컬럼 나누기
num_attribs = [
    "eastsea", 
    "eastsea_anomaly", 
    "mungyeong_temp", 
    "burned_time"    # 🔥 새 이름으로 정확히!
]
cat_attribs = ["month", "day"]
# 🔹 수치형 파이프라인 만들기
num_pipeline = Pipeline([
    ("std_scaler", StandardScaler())
])

# 🔹 전체 파이프라인 만들기
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

# 🔹 strat_train_set 기반 fires
fires = strat_train_set.drop("burned_area", axis=1)
fires_labels = strat_train_set["burned_area"].copy()

# 🔹 training set 변환
fires_prepared = full_pipeline.fit_transform(fires)

# 🔹 test set도 변환
fires_test = strat_test_set.drop("burned_area", axis=1)
fires_test_labels = strat_test_set["burned_area"].copy()

fires_test_prepared = full_pipeline.transform(fires_test)

# 🔹 결과
print("\nfires_prepared 완료")
print(fires_prepared.shape)




# Keras model 개발
X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dense(30, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss="mean_squared_error",
              optimizer=keras.optimizers.SGD(learning_rate=1e-3))

history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))

# Keras 모델 저장
model.save('fires_model.keras')

# evaluate
X_new = X_test[:3]
print("\n","예측 결과\n", np.round(model.predict(X_new), 2))


