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

            input_df = pd.DataFrame([input_data])

            input_prepared = pipeline.transform(input_df)
            pred_log = model.predict(input_prepared)
            pred_final = round(np.expm1(pred_log[0][0]), 2)  # 복원

            # 축구장 몇 개로 변환 (축구장 하나 = 0.714ha)
            soccer_fields = round(pred_final / 0.714, 2)

            return render_template('result.html', prediction=pred_final, soccer_fields=soccer_fields)

        except Exception as e:
            print("❌ Error 발생:", e)
            return "입력값 에러! 다시 확인해주세요."

    return render_template('prediction.html', form=form)


# 서버 시작
if __name__ == '__main__':
    app.run()
