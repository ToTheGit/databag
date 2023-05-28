#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class Preprocessing:
    def __init__(self, df, places):
        self.df = df
        self.places = places
        self.spot = {}
        self.mean = None
        self.std = None

    def preprocess_1(self):
        self.df['PPLTN_TIME'] = pd.to_datetime(self.df['PPLTN_TIME'])
        self.df.loc[:, 'AREA_CONGEST_LVL'] = self.df['AREA_CONGEST_LVL'].replace(
            {'여유': 1, '보통': 2, '약간 붐빔': 3, '붐빔': 4})
        self.df['PRECPT_TYPE'] = self.df['PRECPT_TYPE'].replace('-999', np.nan)
        self.df = self.df.dropna(subset=['PRECPT_TYPE'])
        self.df['PRECPT_TYPE'] = self.df['PRECPT_TYPE'].map({'없음': 0, '비': 1}).astype('category')
        self.df = self.df[['AREA_NM', 'AREA_CONGEST_LVL', 'AREA_PPLTN_MIN', 'AREA_PPLTN_MAX', 'PRECPT_TYPE', 'PPLTN_TIME']]

        for place in self.places:
            self.spot[place] = self.df[self.df['AREA_NM'] == place]
            del self.spot[place]['AREA_NM']

    def preprocess_2(self, place):
        data = self.spot[place].copy()

        data.set_index('PPLTN_TIME', inplace=True)
        new_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='5min')
        data = data.reindex(new_index)

        data['AREA_CONGEST_LVL'] = data['AREA_CONGEST_LVL'].interpolate(method='linear').round()
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
        
        data = pd.get_dummies(data, columns=['PRECPT_TYPE', 'day_of_week', 'is_holiday'])
        
        self.mean = data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].mean(axis=0)
        data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']] -= self.mean
        self.std = data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].std(axis=0)
        data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']] /= self.std
        return data
    
    def preprocess_for_congest_lvl(self, place):
        data = self.spot[place].copy()  # Create a copy to avoid modifying original data
        
        data.set_index('PPLTN_TIME', inplace=True)
        new_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='5min')
        data = data.reindex(new_index)

        data['AREA_CONGEST_LVL'] = data['AREA_CONGEST_LVL'].interpolate(method='linear').round()
        data['AREA_PPLTN_MIN'] = data['AREA_PPLTN_MIN'].interpolate(method='linear').round()
        data['AREA_PPLTN_MAX'] = data['AREA_PPLTN_MAX'].interpolate(method='linear').round()
        data['PRECPT_TYPE'].fillna(method='ffill', inplace=True)  # Forward fill

        data['PRECPT_TYPE'].astype('category')
        data = data.reset_index().rename(columns={'index': 'PPLTN_TIME'})
        
        seq_length = 2016  # one week's data
        num_rows_to_use = len(data) % seq_length
        data = data[num_rows_to_use:]

        holidays = [pd.Timestamp('2023-05-01').date(), pd.Timestamp('2023-05-05').date()]
        data['day_of_week'] = data['PPLTN_TIME'].dt.dayofweek
        data['is_holiday'] = data['PPLTN_TIME'].dt.date.isin(holidays).astype(int)
        data.set_index('PPLTN_TIME', inplace=True)

        data['AREA_CONGEST_LVL'] = data['AREA_CONGEST_LVL'].astype('category')
        data = pd.get_dummies(data, columns=['AREA_CONGEST_LVL', 'PRECPT_TYPE', 'day_of_week', 'is_holiday'])

        self.mean = data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].mean(axis=0)
        data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']] -= self.mean
        self.std = data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']].std(axis=0)
        data[['AREA_PPLTN_MIN', 'AREA_PPLTN_MAX']] /= self.std
        
        return data


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
        if target_column == 'AREA_CONGEST_LVL':
            outputs = layers.Dense(4, activation='softmax')(x)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
        else:
            outputs = layers.Dense(1)(x)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        history = model.fit(dataset, epochs=10)
        return model
            
    def prediction(self, model, sequence_length):
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
            input_sequence[-1, -1] = next_population  # add the prediction to the end of the sequence

            # Save the prediction
            predictions.append(next_population[0, 0])  # save the prediction for this step

        # Convert list to numpy array for easier manipulation
        predictions = np.array(predictions)
        if model_type == 'regression':
            predicted_values = predictions * self.std + self.mean
        else:
            predicted_values = predictions

        return predicted_values

