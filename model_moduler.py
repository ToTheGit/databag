#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import mysql.connector
from datetime import timedelta


# In[3]:


class Preprocessing:
    def __init__(self, df, place):
        self.df = df
        self.place = place
        self.mean = None
        self.std = None
    def filter_data(self): 
        # [~ 일요일 23시 55분]
        self.df['PPLTN_TIME'] = pd.to_datetime(self.df['PPLTN_TIME'])

        # Get the latest date in the dataframe
        last_date = self.df['PPLTN_TIME'].max()
        if last_date.weekday() != 6:
            last_sunday = last_date - timedelta(days=(last_date.weekday() + 1))
        else:
            last_sunday = last_date
        last_sunday = last_sunday.replace(hour=23, minute=55, second=0, microsecond=0)
        # Filter data
        self.df = self.df[self.df['PPLTN_TIME'] <= last_sunday]
    
    def preprocess_1(self): 
        # 결측값 처리, feature 선택 및 preprocess
        self.df['PPLTN_TIME'] = pd.to_datetime(self.df['PPLTN_TIME'])
        self.df['PRECPT_TYPE'] = self.df['PRECPT_TYPE'].replace('-999', np.nan)
        self.df = self.df.dropna(subset=['PRECPT_TYPE'])
        self.df['PRECPT_TYPE'] = self.df['PRECPT_TYPE'].map({'없음': 0, '비': 1}).astype('category')
        self.df = self.df[['AREA_NM', 'AREA_PPLTN_MIN', 'AREA_PPLTN_MAX', 'PRECPT_TYPE', 'PPLTN_TIME']]
        self.df = self.df[self.df['AREA_NM'] == self.place]
        del self.df['AREA_NM']

    def preprocess_2(self, place): 
        # 특정 place fillna, len(data) -> sequence_length 배수, 정규화, 원-핫 인코딩
        # for AREA_PPLTN_MIN, AREA_PPLTN_MAX
        data = self.df.copy()

        data.set_index('PPLTN_TIME', inplace=True)
        # handle duplicate index
        data = data.loc[~data.index.duplicated(keep='first')]
        new_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='5min')
        data = data.reindex(new_index)

#         data['AREA_CONGEST_LVL'] = data['AREA_CONGEST_LVL'].interpolate(method='linear').round()
        data['AREA_PPLTN_MIN'] = data['AREA_PPLTN_MIN'].interpolate(method='linear').round()
        data['AREA_PPLTN_MAX'] = data['AREA_PPLTN_MAX'].interpolate(method='linear').round()
        data['PRECPT_TYPE'].fillna(method='ffill', inplace=True)  # Forward fill

        data['PRECPT_TYPE'].astype('category')
        data = data.reset_index().rename(columns={'index': 'PPLTN_TIME'})

        # Ensure data length is a multiple of sequence_length
        seq_length = 2016  # one week's data
        num_rows_to_use = len(data) % seq_length
        data = data[num_rows_to_use:]

        holidays = [pd.Timestamp('2023-05-01').date(), pd.Timestamp('2023-05-05').date()]
        data['day_of_week'] = data['PPLTN_TIME'].dt.dayofweek
        data['is_holiday'] = data['PPLTN_TIME'].dt.date.isin(holidays).astype(int)
        data.set_index('PPLTN_TIME', inplace=True)
        
        data = pd.get_dummies(data, columns=['PRECPT_TYPE', 'day_of_week', 'is_holiday']).astype('int32')
        
        # Calculate mean and deviation of each row
        data['MEAN'] = data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].mean(axis=1)
        data['DEV'] = data['MEAN'] - data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].min(axis=1)

        # Normalize MEAN and DEV columns
        self.mean = data[['MEAN', 'DEV']].mean()
        self.std = data[['MEAN', 'DEV']].std()

        data[['MEAN', 'DEV']] = (data[['MEAN', 'DEV']] - self.mean) / self.std
        min_max = data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']]
        data = data.drop(['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX'], axis=1)
        return data, min_max


# In[ ]:


class Modeling:
    def __init__(self, data, delay=2304, sequence_length=2016, sampling_rate=1, batch_size=64):
        self.data = data
        self.delay = delay
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size

    def create_dataset(self, target_column):
        dataset = keras.preprocessing.timeseries_dataset_from_array(
            self.data[:-self.delay],
            targets=self.data[target_column][self.delay:],
            sampling_rate=self.sampling_rate,
            sequence_length=self.sequence_length,
            shuffle=False,
            batch_size=self.batch_size,
        )
        return dataset

    def modeling(self, dataset, target_column):
        inputs = keras.Input(shape=(self.sequence_length, self.data.shape[-1]))
        x = layers.GRU(32, return_sequences=True)(inputs)
        x = layers.GRU(32)(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        history = model.fit(dataset, epochs=10)
        return model
            
    def prediction(self, model, sequence_length, target):
        input_sequence = self.data[-sequence_length:].values
        num_features = self.data.shape[1]
        predictions = []  # list to hold our predictions

        # Predict the population for the next 24 hours
        for _ in range(sequence_length):
            # Reshape the sequence to match the model_max's input shape
            input_sequence_reshaped = input_sequence.reshape((1, sequence_length, -1))

            # Make a prediction for the next interval
            next_population = model.predict(input_sequence_reshaped)

            # Append the predicted value to the end of the sequence in the last column
            input_sequence = np.roll(input_sequence, shift=-1, axis=0)  # shift all rows up
            
            input_sequence[-1, -1] = next_population[0, 0]
            # Save the prediction
            predictions.append(next_population[0, 0])  # save the prediction for this step
                
        # Convert list to numpy array for easier manipulation
        predictions = np.array(predictions)
        predicted_values = predictions * preprocessor.std[target] + preprocessor.mean[target]
        return predicted_values
    
    def get_ready(predicted_values, min_max):
        df_predictions = pd.DataFrame(predicted_values)
        df_predictions['AREA_PPLTN_MAX'] = df_predictions['MEAN'] + df_predictions['DEV']
        df_predictions['AREA_PPLTN_MIN'] = df_predictions['MEAN'] - df_predictions['DEV']
        df_predictions = df_predictions.drop(['MEAN', 'DEV'], axis=1)
        
        min_max['weekday'] = min_max.index.weekday
        min_max['time'] = min_max.index.time
        min_max_grouped = min_max.groupby(['weekday', 'time']).agg(
            {'AREA_PPLTN_MIN': 'min', 'AREA_PPLTN_MAX': 'max'}
        )
        min_max_grouped.reset_index(inplace=True)
        min_max_grouped = min_max_grouped.drop(['weekday', 'time'], axis=1)
        
        # First, compute the averages for each DataFrame
        df_predictions['AVG'] = df_predictions[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].mean(axis=1)
        min_max_grouped['AVG'] = min_max_grouped[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].mean(axis=1)
        # Make sure that the indexes align before computing the percentages
        df_predictions = df_predictions.sort_index()
        min_max_grouped = min_max_grouped.sort_index()
        # Compute the percentages and store them in a new column in df_predictions
        df_predictions['AREA_CONGEST_PER'] = (df_predictions['AVG'] / min_max_grouped['AVG']) * 100
        
        df_predictions[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX', 'AREA_CONGEST_PER']] = df_predictions[[
    'AREA_PPLTN_MIN', 'AREA_PPLTN_MAX', 'AREA_CONGEST_PER']].astype('int64')
        return df_predictions
    
    def extract_representative_values(data):
        functions = {
            'AREA_CONGEST_PER': max,
            'AREA_PPLTN_MIN': min,
            'AREA_PPLTN_MAX': max
        }

        for col, func in functions.items():
            data[col] = [func(data[col].values[i:i+12]) for i in range(0, len(data), 12)]
        return data


# In[ ]:


class DatabaseUpdater:
    def __init__(self, user, password, host, database):
        self.cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self.cursor = self.cnx.cursor()

    def to_sql(df_predictions, place):
        df_predictions = df_predictions.reset_index(drop=True)
        df_predictions['id'] = df_predictions.index
        df_predictions['AREA_NM'] = place
        # Connect to the database
        cnx = mysql.connector.connect(user='dbid231', password='dbpass231',
                                      host='localhost',
                                      database='db23103')
        cursor = cnx.cursor()

        for i, row in df_predictions.iterrows():
            # Create the SQL INSERT command with ON DUPLICATE KEY UPDATE
            insert_query = f"""
            INSERT INTO predict_congestions_tb
            (id, AREA_NM, AREA_CONGEST_PER, AREA_PPLTN_MIN, AREA_PPLTN_MAX) 
            VALUES ({i}, '{place}', {row['AREA_CONGEST_PER']}, {row['AREA_PPLTN_MIN']}, {row['AREA_PPLTN_MAX']})
            ON DUPLICATE KEY UPDATE
            AREA_CONGEST_PER = VALUES(AREA_CONGEST_PER),
            AREA_PPLTN_MIN = VALUES(AREA_PPLTN_MIN),
            AREA_PPLTN_MAX = VALUES(AREA_PPLTN_MAX)
            """

            # Execute the SQL command
            cursor.execute(insert_query)

        # Commit changes and close connection
        cnx.commit()
        cursor.close()
        cnx.close()


# In[ ]:





# In[ ]:





# In[ ]:




