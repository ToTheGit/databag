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


# In[ ]:

start_time = time.time()
df = pd.read_csv("seoul_citydata_2 (16).csv")
end_time = time.time()
elapsed_time = end_time - start_time
print("df load 소요 시간: {:.2f} seconds".format(elapsed_time))

# In[ ]:


place = '여의도'


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
modeling_mean = Modeling(mean_dev)
dataset_mean = modeling_mean.create_dataset('MEAN')
model_mean = modeling_mean.modeling(dataset_mean, 'MEAN')
sequence_length_ppltn = 2016  # one week's data
predicted_values['MEAN'] = modeling_mean.prediction(model_mean, sequence_length_ppltn, 'MEAN')
end_time = time.time()
elapsed_time = end_time - start_time
print("MEAN 예측완료 \n")
print("소요 시간: {:.2f} seconds".format(elapsed_time))


# In[ ]:

start_time = time.time()
modeling_dev = Modeling(mean_dev)
dataset_dev = modeling_dev.create_dataset('DEV')
model_dev = modeling_dev.modeling(dataset_dev, 'DEV')
sequence_length_ppltn = 2016  # one week's data
predicted_values['DEV'] = modeling_mean.prediction(model_dev, sequence_length_ppltn, 'DEV')
print("AREA_PPLTN_MIN 예측완료")
elapsed_time = end_time - start_time
print("DEV 예측완료 \n")
print("df load 소요 시간: {:.2f} seconds".format(elapsed_time))


# In[ ]:


df_predictions = get_ready(df_predictions, min_max)
print("get_ready 완료")


# In[ ]:


extract = Modeling(df_predictions)
df_predictions = extract.extract_representative_values()
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
