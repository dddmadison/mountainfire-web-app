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


# 1-1 불러오기
fires = pd.read_csv("./data/sanbul2district-divby100.csv")
fires["burned_area"] = np.log1p(fires["burned_area"])

# 1-2 기본 정보 확인
print("2019210147 우상용")
print("This is head")
print(fires.head())
print("This is info")
print(fires.info())
print("This is describe")
print(fires.describe())
print("This is Month count")
print(fires["month"].value_counts())
print("This is Day count")
print(fires["day"].value_counts())


# 전체 수치형 변수 히스토그램 1-3
fires.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
# plt.show()


#1-3 추가
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 시각화 스타일 지정 (선택)
# sns.set(style="whitegrid")

# # 🔹 1. 'burned_area' 분포 확인 (로그 변환 후)
# plt.figure(figsize=(8, 4))
# sns.histplot(fires["burned_area"], kde=True, bins=40)
# plt.title("Distribution of Burned Area (log-scaled)")
# plt.xlabel("Log(Burned Area + 1)")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()

# # 🔹 2. 'avg_temp' vs 'burned_area' 산점도
# plt.figure(figsize=(8, 4))
# sns.scatterplot(data=fires, x="avg_temp", y="burned_area", alpha=0.6)
# plt.title("Avg Temperature vs Burned Area")
# plt.xlabel("Average Temperature (°C)")
# plt.ylabel("Log(Burned Area + 1)")
# plt.tight_layout()
# plt.show()

# # 🔹 3. 월별 평균 소실면적 (카테고리 분석)
# plt.figure(figsize=(10, 4))
# sns.boxplot(data=fires, x="month", y="burned_area")
# plt.title("Burned Area by Month")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


#1-4
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
fires_raw = pd.read_csv("./data/sanbul2district-divby100.csv")
plt.hist(fires_raw["burned_area"], bins=30, color="skyblue", edgecolor="black")
plt.title("Before log (burned_area)")
plt.xlabel("burned_area")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(fires["burned_area"], bins=30, color="salmon", edgecolor="black")
plt.title("After log1p(burned_area)")
plt.xlabel("log(1 + burned_area)")

plt.tight_layout()
# plt.show()

#1-5

# 🔹 임시로 일반 분할도 진행
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

# 🔹 month 분포 시각화
fires["month"].hist()

# 🔹 StratifiedShuffleSplit로 month 기준 stratified 분할
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

# 🔹 분포 출력
print("\nMonth category proportion (strat_test_set):\n",
      strat_test_set["month"].value_counts() / len(strat_test_set))

print("\nOverall month category proportion (전체 fires):\n",
      fires["month"].value_counts() / len(fires))



#1-6
# fires_full = pd.read_csv("./data/sanbul2district-divby100.csv")
# fires["burned_area"] = np.log1p(fires["burned_area"])

# split target/feature
X = fires.drop("burned_area", axis=1)
y = fires["burned_area"]

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # OneHotEncoder (여기서부터 인코딩 시작!)
# categorical_cols = ["month", "day"]
# encoder = OneHotEncoder(sparse_output=False)

# encoded = encoder.fit_transform(X_train[categorical_cols])
# encoded_cols = encoder.get_feature_names_out(categorical_cols)
# encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X_train.index)

# X_train_encoded = pd.concat([X_train.drop(categorical_cols, axis=1), encoded_df], axis=1)

# # X_test도 동일하게
# encoded_test = encoder.transform(X_test[categorical_cols])
# encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)
# X_test_encoded = pd.concat([X_test.drop(categorical_cols, axis=1), encoded_test_df], axis=1)

# # 확인
# print(f"X_train_encoded shape: {X_train_encoded.shape}")
# print(f"X_test_encoded shape: {X_test_encoded.shape}")
# print(f"Encoded columns: {list(encoded_cols)}")

# # 🔹 시각화 대상 수치형 변수 5개 (원하는 만큼 늘릴 수 있음)
# selected_features = ["avg_temp", "max_temp", "max_wind_speed", "avg_wind", "burned_area"]

# # 🔹 산점도 행렬 출력
# scatter_matrix(fires[selected_features], figsize=(12, 8), alpha=0.7, diagonal="hist")

# # 제목 등 설정
# plt.suptitle("Scatter Matrix of Fire-related Features", fontsize=16)
# plt.tight_layout()
# plt.show()

#1-7
fires.plot(kind="scatter",
           x="longitude",
           y="latitude",
           alpha=0.4,
           s=fires["max_temp"],          # 원 크기
           c="burned_area",              # 컬러
           cmap=plt.get_cmap("jet"),     # jet 컬러맵 사용
           colorbar=True,
           figsize=(10, 6))

plt.title("Burned Area by Location\n(s: max_temp, c: burned_area)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
# plt.show()

#1-8

# 🔹 CSV 읽기 및 log 변환
fires_full = pd.read_csv("./data/sanbul2district-divby100.csv")
fires_full["burned_area"] = np.log1p(fires_full["burned_area"])

# 🔹 Stratified split 준비 (month 기준)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires_full, fires_full["month"]):
    strat_train_set = fires_full.loc[train_index]
    strat_test_set = fires_full.loc[test_index]

# 🔹 OneHotEncoding 작업 (이제 strat_train_set 사용 가능!)
fires = strat_train_set.drop("burned_area", axis=1)
fires_labels = strat_train_set["burned_area"].copy()

fires_num = fires.drop(["month", "day"], axis=1)
cat_attrs = ["month", "day"]

encoder = OneHotEncoder(sparse_output=False)
cat_encoded = encoder.fit_transform(fires[cat_attrs])

encoded_cols = encoder.get_feature_names_out(cat_attrs)
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoded_cols, index=fires.index)

fires_prepared = pd.concat([fires_num, cat_encoded_df], axis=1)

# 🔹 출력
print("OneHot Encoding된 컬럼:")
print(list(encoded_cols))
print("\n인코딩된 DataFrame 샘플:")
print(fires_prepared.head())


# # 1-9

# 🔹 수치형, 범주형 컬럼 나누기
num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
cat_attribs = ["month", "day"]

# 🔹 수치형 파이프라인
num_pipeline = Pipeline([
    ("std_scaler", StandardScaler())
])

# 🔹 전체 파이프라인 (ColumnTransformer)
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

# 🔹 strat_train_set 기반 fires (burned_area 제외)
fires = strat_train_set.drop("burned_area", axis=1)

# 🔹 변환 적용
fires_prepared = full_pipeline.fit_transform(fires)
# strat_test_set도 변환
fires_test = strat_test_set.drop("burned_area", axis=1)
fires_test_labels = strat_test_set["burned_area"].copy()

fires_test_prepared = full_pipeline.transform(fires_test)

print("\nfires_prepared 완료")
print(fires_prepared.shape)


# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

# 랜덤 시드 고정 (재현성)
np.random.seed(42)
tf.random.set_seed(42)

# Keras 모델 정의
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[X_train.shape[1]]),
    keras.layers.Dense(30, activation="relu", input_shape=[X_train.shape[1]]),
    keras.layers.Dense(30, activation="relu", input_shape=[X_train.shape[1]]),
    keras.layers.Dense(1)   # 출력층 (회귀니까 1개)
])

# 모델 구조 출력
model.summary()

# 컴파일
model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)

# 학습
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_valid, y_valid)
)

# 모델 저장
model.save("fires_model.keras")

# 평가
X_new = X_test[:3]
print("\n▶ 예측 결과:\n", np.round(model.predict(X_new), 2))


joblib.dump(full_pipeline, "app/models/full_pipeline.pkl")