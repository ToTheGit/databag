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
        
        self.mean = data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].mean(axis=0)
        data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']] -= self.mean
        self.std = data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].std(axis=0)
        data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']] /= self.std
        
        target1 = data['AREA_PPLTN_MIN'].values
        target2 = data['AREA_PPLTN_MAX'].values
        targets = np.stack((target1, target2), axis=-1)
        return data, targets


# In[ ]:


class Modeling:
    def __init__(self, preprocessor, data, targets, delay=2304, sequence_length=2016, sampling_rate=1, batch_size=64):
        self.preprocessor = preprocessor
        self.data = data
        self.delay = delay
        self.targets = targets
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size

    def create_dataset(self):
        dataset = keras.preprocessing.timeseries_dataset_from_array(
            self.data[:-self.delay],
            targets= self.targets[self.delay:],
            sampling_rate=self.sampling_rate,
            sequence_length=self.sequence_length,
            shuffle=False,
            batch_size=self.batch_size,
        )
        return dataset

    def modeling(self, dataset):
        inputs = keras.Input(shape=(self.sequence_length, self.data.shape[-1]))
        # Shared layers
        x = layers.GRU(32, return_sequences=True)(inputs)
        x = layers.GRU(32)(x)
        # Output layers
        output1 = layers.Dense(1, name='output1')(x)  # Output layer for first target
        output2 = layers.Dense(1, name='output2')(x)  # Output layer for second target
        model = keras.Model(inputs, [output1, output2])
        model.compile(optimizer="Adam", loss={"output1": "mse", "output2": "mse"}, 
                      metrics={"output1": ["mae"], "output2": ["mae"]}, )
        history = model.fit(dataset, epochs=20)
        return model
            
    def prediction(self, model, sequence_length):
        input_sequence = self.data[-sequence_length:].values
        num_features = self.data.shape[1]
        predictions1 = []  # list to hold our predictions for output1
        predictions2 = []  # list to hold our predictions for output2

        # Predict for the next 24 hours
        for _ in range(sequence_length):
            # Reshape the sequence to match the model's input shape
            input_sequence_reshaped = input_sequence.reshape((1, sequence_length, -1))

            # Make a prediction for the next interval
            next_output = model.predict(input_sequence_reshaped)

            # Append the predicted value to the end of the sequence in the last column
            input_sequence = np.roll(input_sequence, shift=-1, axis=0)  # shift all rows up

            input_sequence[-1, -1] = next_output[0]  # this should be adjusted based on the structure of your data
            input_sequence[-1, -2] = next_output[1]  # this should be adjusted based on the structure of your data

            # Save the prediction
            predictions1.append(next_output[0])  # save the prediction for output1
            predictions2.append(next_output[1])  # save the prediction for output2

        # Convert lists to numpy arrays for easier manipulation
        predictions1 = np.array(predictions1)
        predictions2 = np.array(predictions2)
        
        predictions1 = predictions1 * self.preprocessor.std['AREA_PPLTN_MIN'] + self.preprocessor.mean['AREA_PPLTN_MIN']
        predictions2 = predictions2 * self.preprocessor.std['AREA_PPLTN_MAX'] + self.preprocessor.mean['AREA_PPLTN_MAX'] 
                # Reshape the arrays
        predictions1 = predictions1.reshape(-1)
        predictions2 = predictions2.reshape(-1)
        # Create dataframes
        df_predictions1 = pd.DataFrame(predictions1, columns=['AREA_PPLTN_MIN'])
        df_predictions2 = pd.DataFrame(predictions2, columns=['AREA_PPLTN_MAX'])
        # If you want to combine these two dataframes into one, you can do:
        df_predictions = pd.concat([df_predictions1, df_predictions2], axis=1)
        return df_predictions
    
    def get_ready(self, df_predictions):
        def swap_min_max(row):
            if row['AREA_PPLTN_MIN'] > row['AREA_PPLTN_MAX']:
                row['AREA_PPLTN_MIN'], row['AREA_PPLTN_MAX'] = row['AREA_PPLTN_MAX'], row['AREA_PPLTN_MIN']
            return row
        df_predictions = df_predictions.apply(swap_min_max, axis=1)
        
        df_predictions[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']] = df_predictions[[
    'AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].astype('int64')
        return df_predictions
    
    def extract_representative_values(self, data):
        data['group'] = np.arange(len(data)) // 12

        def custom_operations(group):
            return pd.Series({
                'AREA_PPLTN_MIN': group['AREA_PPLTN_MIN'].min(),
                'AREA_PPLTN_MAX': group['AREA_PPLTN_MAX'].max()
            })
        # Group by the 'group' column and apply your custom operations
        df_grouped = data.groupby('group').apply(custom_operations)
        # Reset the index if you want
        # Step 1: Calculate the global minimum and maximum values
        global_min = df_grouped['AREA_PPLTN_MIN'].min()
        global_max = df_grouped['AREA_PPLTN_MAX'].max()

        # Step 2: Calculate the percentage of `AREA_PPLTN_MAX` relative to the range of the global maximum and minimum values
        df_grouped['AREA_CONGEST_LVL'] = ((df_grouped['AREA_PPLTN_MAX'] - global_min) / (global_max - global_min)) * 100
        
        # Define the bins
        bins = [-np.inf, 33, 50, 75, np.inf]
        # Define the labels for the bins
        labels = ['여유', '보통', '약간 붐빔', '붐빔']
        # Create a new column with the category data
        df_grouped['AREA_CONGEST_LVL'] = pd.cut(df_grouped['AREA_CONGEST_LVL'], bins=bins, labels=labels)
        
        df_grouped.reset_index(drop=True, inplace=True)
        return df_grouped


# In[ ]:


class DatabaseUpdater:
    def __init__(self, user, password, host, database):
        self.cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        self.cursor = self.cnx.cursor()

    def to_sql(self, df_predictions, place, places):
        df_predictions = df_predictions.reset_index(drop=True)
        df_predictions['id'] = df_predictions.index
        place_index = places.index(place)
        df_predictions['AREA_NM'] = place_index
        # Connect to the database
        cnx = mysql.connector.connect(user='dbid231', password='dbpass231',
                                      host='localhost',
                                      database='db23103')
        cursor = cnx.cursor()

        for i, row in df_predictions.iterrows():
            # Create the SQL INSERT command with ON DUPLICATE KEY UPDATE
            insert_query = f"""
            INSERT INTO congestions_tb
            (id, AREA_NM, AREA_CONGEST_LVL, AREA_PPLTN_MIN, AREA_PPLTN_MAX) 
            VALUES ({i}, '{place}', '{row['AREA_CONGEST_LVL']}', {row['AREA_PPLTN_MIN']}, {row['AREA_PPLTN_MAX']})
            ON DUPLICATE KEY UPDATE
            AREA_CONGEST_LVL = VALUES(AREA_CONGEST_LVL),
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




