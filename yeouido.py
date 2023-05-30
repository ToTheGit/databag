#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import mysql.connector
from datetime import timedelta
import time
from model_moduler import Preprocessing, Modeling, DatabaseUpdater
import multiprocessing


def run_model(place):
    print(place, "예측")
    start_time = time.time()
    df = pd.read_csv("seoul_citydata_2 (16).csv")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("df load 소요 시간: {:.2f} seconds".format(elapsed_time))
    # In[ ]:
#     place = '여의도'
    # In[ ]:
    preprocessor = Preprocessing(df, place)
    # In[ ]:
    preprocessor.filter_data()
    print("filter_data 완료")
    # In[ ]:
    preprocessor.preprocess_1()
    print("preprocess_1 완료")
    # In[ ]:
    mean_dev, min_max = preprocessor.preprocess_2(place)
    print("preprocess_2 완료")
    # In[ ]:
    predicted_values = {}
    # In[ ]:
    start_time = time.time()
    modeling_mean = Modeling(preprocessor, mean_dev)
    dataset_mean = modeling_mean.create_dataset('MEAN')
    model_mean = modeling_mean.modeling(dataset_mean, 'MEAN')
    sequence_length_ppltn = 2016  # one week's data
    elapsed_time = end_time - start_time
    print("MEAN 훈련 완료 \n")
    print("MEAN 훈련 시간: {:.2f} seconds".format(elapsed_time))

    start_time = time.time()
    predicted_values['MEAN'] = modeling_mean.prediction(model_mean, sequence_length_ppltn, 'MEAN')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("MEAN 예측 완료 \n")
    print("MEAN 예측 시간: {:.2f} seconds".format(elapsed_time))
    # In[ ]:
    start_time = time.time()
    modeling_dev = Modeling(preprocessor, mean_dev)
    dataset_dev = modeling_dev.create_dataset('DEV')
    model_dev = modeling_dev.modeling(dataset_dev, 'DEV')
    sequence_length_ppltn = 2016  # one week's data
    elapsed_time = end_time - start_time
    print("DEV 훈련 완료 \n")
    print("DEV 훈련 시간: {:.2f} seconds".format(elapsed_time))

    start_time = time.time()
    predicted_values['DEV'] = modeling_dev.prediction(model_dev, sequence_length_ppltn, 'DEV')
    elapsed_time = end_time - start_time
    print("DEV 예측 완료 \n")
    print("DEV 예측 시간: {:.2f} seconds".format(elapsed_time))
    # In[ ]:
    ready = Modeling(preprocessor, predicted_values)
    df_predictions = ready.get_ready(predicted_values, min_max)
    print("get_ready 완료")
    # In[ ]:
    df_predictions = ready.extract_representative_values(df_predictions)
    print("extract_representative_values 완료")
    # In[ ]:
    updater = DatabaseUpdater(user='dbid231', password='dbpass231', host='localhost', database='db23103')
    # In[ ]:
    start_time = time.time()
    print("DB저장 시작")
    updater.to_sql(df_predictions, place)
    elapsed_time = end_time - start_time
    print("df load 소요 시간: {:.2f} seconds".format(elapsed_time))
    print("DB저장 완료")
if __name__ == "__main__":
    places = ['강남 MICE 관광특구',
 '동대문 관광특구',
 '명동 관광특구',
 '이태원 관광특구',
 '잠실 관광특구',
 '종로·청계 관광특구',
 '홍대 관광특구',
 '경복궁·서촌마을',
 '광화문·덕수궁',
 '창덕궁·종묘',
 '가산디지털단지역',
 '강남역',
 '건대입구역',
 '고속터미널역',
 '교대역',
 '구로디지털단지역',
 '서울역',
 '선릉역',
 '신도림역',
 '신림역',
 '신촌·이대역',
 '역삼역',
 '연신내역',
 '용산역',
 '왕십리역',
 'DMC(디지털미디어시티)',
 '창동 신경제 중심지',
 '노량진',
 '낙산공원·이화마을',
 '북촌한옥마을',
 '가로수길',
 '성수카페거리',
 '수유리 먹자골목',
 '쌍문동 맛집거리',
 '압구정로데오거리',
 '여의도',
 '영등포 타임스퀘어',
 '인사동·익선동',
 '국립중앙박물관·용산가족공원',
 '남산공원',
 '뚝섬한강공원',
 '망원한강공원',
 '반포한강공원',
 '북서울꿈의숲',
 '서울대공원',
 '서울숲공원',
 '월드컵공원',
 '이촌한강공원',
 '잠실종합운동장',
 '잠실한강공원']
    num_processes = 10
    with multiprocessing.Pool(num_processes) as p:
        p.map(run_model, places)