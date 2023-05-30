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
from datetime import timedelta, datetime
import time
from model_moduler import Preprocessing, Modeling, DatabaseUpdater
import multiprocessing

def main(place):
    # print(place, "예측")
    start_time = time.time()
    df = pd.read_csv("seoul_citydata_2 (16).csv")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("df load 소요 시간: {:.2f} seconds".format(elapsed_time))
    # In[ ]:
    # place = '여의도'
    # In[ ]:
    preprocessor = Preprocessing(df, place)
    # In[ ]:
    preprocessor.filter_data()
    print("filter_data 완료")
    # In[ ]:
    preprocessor.preprocess_1()
    print("preprocess_1 완료")
    # In[ ]:
    data, targets = preprocessor.preprocess_2(place)
    print("preprocess_2 완료")
    # In[ ]:
    # In[ ]:

    print("Model 훈련 시작", datetime.now())
    modeling = Modeling(preprocessor, data, targets)
    dataset = modeling.create_dataset()
    model = modeling.modeling(dataset)
    sequence_length_ppltn = 2016  # one week's data
    print("Model 훈련 완료", datetime.now())

    print("Model 예측 시작", datetime.now())
    start_time = time.time()
    predicted_values = modeling.prediction(model, sequence_length_ppltn)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Model 예측 완료", datetime.now())


    # In[ ]:
    ready = Modeling(preprocessor, predicted_values, targets)
    df_predictions = ready.get_ready(predicted_values)
    print("get_ready 완료")
    # In[ ]:
    df_predictions = ready.extract_representative_values(df_predictions)
    print("extract_representative_values 완료")
    # In[ ]:
    updater = DatabaseUpdater(user='dbid231', password='dbpass231', host='localhost', database='db23103')
    # In[ ]:

    print("DB저장 시작", datetime.now())
    updater.to_sql(df_predictions, place)
    elapsed_time = end_time - start_time
    print("DB저장 완료", datetime.now())
