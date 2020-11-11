#!/usr/bin/env python
# coding: utf-8
# %%
from keras.models import Sequential
from keras.layers import Dense


# %%
def baseline_model():
    model = Sequential()
    model.add(Dense(12, input_dim=110, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(9, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# %%
baseline_model()

