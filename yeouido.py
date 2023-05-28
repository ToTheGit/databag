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


# In[ ]:


df = pd.read_csv("seoul_citydata_2 (16).csv")


# In[ ]:


place = '여의도'


# In[ ]:


preprocessor = Preprocessing(df, place)


# In[ ]:


preprocessor.filter_data()


# In[ ]:


preprocessor.preprocess_1()


# In[ ]:


preprocessor.df


# In[ ]:


mean_dev, min_max = preprocessor.preprocess_2(place)


# In[ ]:


predicted_values = {}


# In[ ]:


modeling_mean = Modeling(mean_dev)
dataset_mean = modeling_mean.create_dataset('MEAN')
model_mean = modeling_mean.modeling(dataset_mean, 'MEAN')
sequence_length_ppltn = 2016  # one week's data
predicted_values['MEAN'] = modeling_mean.prediction(model_mean, sequence_length_ppltn, 'MEAN')
print("AREA_PPLTN_MIN 예측완료")


# In[ ]:


modeling_dev = Modeling(mean_dev)
dataset_dev = modeling_dev.create_dataset('DEV')
model_dev = modeling_dev.modeling(dataset_dev, 'DEV')
sequence_length_ppltn = 2016  # one week's data
predicted_values['DEV'] = modeling_mean.prediction(model_dev, sequence_length_ppltn, 'DEV')
print("AREA_PPLTN_MIN 예측완료")


# In[ ]:


df_predictions = get_ready(df_predictions, min_max)


# In[ ]:


extract = Modeling(df_predictions)
df_predictions = extract.extract_representative_values()


# In[ ]:


updater = DatabaseUpdater(user='dbid231', password='dbpass231', host='localhost', database='db23103')


# In[ ]:


updater.to_sql(df_predictions, place)

