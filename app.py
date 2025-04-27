from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import joblib

# Flask 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)

# 입력 폼 클래스
class LabForm(FlaskForm):
    eastsea = StringField('동해 수온', validators=[DataRequired()])
    westsea_anomaly = StringField('서해 해수면 온도 이상', validators=[DataRequired()])
    eastsea_anomaly = StringField('동해 해수면 온도 이상', validators=[DataRequired()])
    eastchina_anomaly = StringField('동중국해 해수면 온도 이상', validators=[DataRequired()])
    eastasia_anomaly = StringField('동아시아 해수면 온도 이상', validators=[DataRequired()])
    mungyeong_temp = StringField('문경 평균 기온', validators=[DataRequired()])
    submit = SubmitField('Submit')

# 메인 페이지
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# 예측 페이지
@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        try:
            model = keras.models.load_model("fires_model.keras")
            pipeline = joblib.load("models/full_pipeline.pkl")

            input_data = {
                'eastsea': float(form.eastsea.data),
                'westsea_anomaly': float(form.westsea_anomaly.data),
                'eastsea_anomaly': float(form.eastsea_anomaly.data),
                'eastchina_anomaly': float(form.eastchina_anomaly.data),
                'eastasia_anomaly': float(form.eastasia_anomaly.data),
                'mungyeong_temp': float(form.mungyeong_temp.data)
            }

            # 입력 데이터프레임 생성
            input_df = pd.DataFrame([input_data])
            print("✅ input_df 생성됨:")
            print(input_df)

            # 변환
            input_prepared = pipeline.transform(input_df)
            print("✅ input_prepared (전처리 후):")
            print(input_prepared)

            # 예측
            prediction = model.predict(input_prepared)
            print("✅ model.predict 결과:")
            print(prediction)

            # NaN 체크
            if np.isnan(prediction[0][0]):
                print("❌ 예측값이 NaN입니다! 입력값 문제 or 모델 문제!")
                return render_template('result.html', prediction="예측할 수 없습니다. 입력을 다시 확인해주세요.")

            # 정상 출력
            pred_final = round(prediction[0][0], 2)
            print("✅ 최종 예측 결과:", pred_final)

            return render_template('result.html', prediction=pred_final)

        except Exception as e:
            print("❌ 예외 발생:", e)
            return "입력값 에러! 다시 확인해주세요."

    return render_template('prediction.html', form=form)


# 서버 시작
if __name__ == '__main__':
    app.run()
