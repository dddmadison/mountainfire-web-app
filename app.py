# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pandas as pd
# import joblib

# from flask import Flask, render_template, request
# from flask_bootstrap import Bootstrap
# from flask_wtf import FlaskForm
# from wtforms import StringField, SubmitField
# from wtforms.validators import DataRequired

# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# # 🔹 Flask 초기화
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'hard to guess string'
# bootstrap = Bootstrap(app)

# # 🔹 입력 폼 클래스
# class LabForm(FlaskForm):
#     longitude = StringField('longitude(1-7)', validators=[DataRequired()])
#     latitude = StringField('latitude(1-7)', validators=[DataRequired()])
#     month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
#     day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
#     avg_temp = StringField('avg_temp', validators=[DataRequired()])
#     max_temp = StringField('max_temp', validators=[DataRequired()])
#     max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
#     avg_wind = StringField('avg_wind', validators=[DataRequired()])
#     submit = SubmitField('Submit')

# # 🔹 모델과 파이프라인 로드
# # model = keras.models.load_model("../fires_model.keras")
# # pipeline = joblib.load("models/full_pipeline.pkl")

# #랜더 호스팅용
# pipeline = joblib.load("models/full_pipeline.pkl")

# model = keras.models.load_model("fires_model.keras")

# # 🔹 라우팅
# @app.route('/')
# @app.route('/index')
# def index():
#     return render_template('index.html')

# @app.route('/prediction', methods=['GET', 'POST'])
# def lab():
#     form = LabForm()
#     if form.validate_on_submit():
#         try:
#             print("✅ 폼 데이터 수신 완료")
#             print("longitude:", form.longitude.data)
#             print("latitude:", form.latitude.data)
#             print("month:", form.month.data)
#             print("day:", form.day.data)
#             print("avg_temp:", form.avg_temp.data)
#             print("max_temp:", form.max_temp.data)
#             print("max_wind_speed:", form.max_wind_speed.data)
#             print("avg_wind:", form.avg_wind.data)

#             # month와 day 변환 맵
#             month_map = {
#                 1: '01-Jan', 2: '02-Feb', 3: '03-Mar', 4: '04-Apr',
#                 5: '05-May', 6: '06-Jun', 7: '07-Jul', 8: '08-Aug',
#                 9: '09-Sep', 10: '10-Oct', 11: '11-Nov', 12: '12-Dec'
#             }
#             day_map = {
#                 0: '00-sun', 1: '01-mon', 2: '02-tue', 3: '03-wed',
#                 4: '04-thu', 5: '05-fri', 6: '06-sat', 7: '07-hol'
#             }

#             # 변환
#             month_input = month_map.get(int(form.month.data))
#             day_input = day_map.get(int(form.day.data))

#             # 입력 데이터 구성
#             input_data = {
#                 'longitude': float(form.longitude.data),
#                 'latitude': float(form.latitude.data),
#                 'month': month_input,
#                 'day': day_input,
#                 'avg_temp': float(form.avg_temp.data),
#                 'max_temp': float(form.max_temp.data),
#                 'max_wind_speed': float(form.max_wind_speed.data),
#                 'avg_wind': float(form.avg_wind.data)
#             }

#             # 정확한 순서로 DataFrame 만들기
#             input_df = pd.DataFrame([input_data], columns=[
#                 'longitude', 'latitude', 'month', 'day', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'
#             ])

#             print("✅ input_df 생성 성공")
#             print(input_df)

#             input_prepared = pipeline.transform(input_df)
#             prediction = round(np.expm1(model.predict(input_prepared)[0][0]), 2)

#             return render_template('result.html', prediction=prediction)

#         except Exception as e:
#             print("❌ Error 발생:", e)
#             return "입력값 에러! 다시 확인해주세요."

#     return render_template('prediction.html', form=form)





# if __name__ == '__main__':
#     app.run(debug=True)


############################여기는 까지 로컬용###############


# 🔹 기본 라이브러리
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 🔹 Flask 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)

# 🔹 입력 폼 클래스
class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

# 🔹 (★랜더용★) 모델과 파이프라인은 서버 시작할 때 메모리에 올리지 않는다.
# pipeline = joblib.load("models/full_pipeline.pkl")
# model = keras.models.load_model("fires_model.keras")

# 🔹 라우팅
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        try:
            print("✅ 폼 데이터 수신 완료")
            print("longitude:", form.longitude.data)
            print("latitude:", form.latitude.data)
            print("month:", form.month.data)
            print("day:", form.day.data)
            print("avg_temp:", form.avg_temp.data)
            print("max_temp:", form.max_temp.data)
            print("max_wind_speed:", form.max_wind_speed.data)
            print("avg_wind:", form.avg_wind.data)

            # 🔥 (★랜더용★) 폼 데이터 받으면 그때 모델/파이프라인 불러오기
            model = keras.models.load_model("fires_model.keras")
            pipeline = joblib.load("models/full_pipeline.pkl")

            # month와 day 변환 맵
            month_map = {
                1: '01-Jan', 2: '02-Feb', 3: '03-Mar', 4: '04-Apr',
                5: '05-May', 6: '06-Jun', 7: '07-Jul', 8: '08-Aug',
                9: '09-Sep', 10: '10-Oct', 11: '11-Nov', 12: '12-Dec'
            }
            day_map = {
                0: '00-sun', 1: '01-mon', 2: '02-tue', 3: '03-wed',
                4: '04-thu', 5: '05-fri', 6: '06-sat', 7: '07-hol'
            }

            # 변환
            month_input = month_map.get(int(form.month.data))
            day_input = day_map.get(int(form.day.data))

            # 입력 데이터 구성
            input_data = {
                'longitude': float(form.longitude.data),
                'latitude': float(form.latitude.data),
                'month': month_input,
                'day': day_input,
                'avg_temp': float(form.avg_temp.data),
                'max_temp': float(form.max_temp.data),
                'max_wind_speed': float(form.max_wind_speed.data),
                'avg_wind': float(form.avg_wind.data)
            }

            # 정확한 순서로 DataFrame 만들기
            input_df = pd.DataFrame([input_data], columns=[
                'longitude', 'latitude', 'month', 'day', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'
            ])

            print("✅ input_df 생성 성공")
            print(input_df)

            # 변환 및 예측
            input_prepared = pipeline.transform(input_df)
            prediction = round(np.expm1(model.predict(input_prepared)[0][0]), 2)

            return render_template('result.html', prediction=prediction)

        except Exception as e:
            print("❌ Error 발생:", e)
            return "입력값 에러! 다시 확인해주세요."

    return render_template('prediction.html', form=form)

# 🔹 서버 시작
if __name__ == '__main__':
    app.run(debug=True)
