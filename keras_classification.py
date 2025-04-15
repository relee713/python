# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)

df = pd.read_excel("C:\\Users\\relee\\Downloads\\wine quality combined.xlsx")
wine = pd.get_dummies(df, columns=["quality", "category"], drop_first=True)
quality_cols = ["quality_4", "quality_5", "quality_6", "quality_7", "quality_8", "quality_9"]

X = wine.drop(columns=quality_cols)
y = wine[quality_cols]

#Prep train vs. test (50/50)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5) 

#CREATE!

model = Sequential()
model.add(Dense(15, activation='tanh'))         #hyperbolic tangent, outputs -1~1
model.add(Dense(10, activation = 'sigmoid'))    #linear regression, binary output
model.add(Dense(20, activation='relu'))         #outputs x < 0 = 0 | x > 0 = x 
model.add(Dense(6, activation='softmax'))       #one-hot code, outputs classification probability 

model.summary()

#COMPILE!
    #loss = CategoricalCrossentropy(): one-hot encoded/categorical problem
    #loss = 'mse': regression problems 
    
    #optimizer = SGD(): constant learning rate and momentum 
    #optimizer = 'adagrad': custom learning via cumulative sum
    #optimizer = 'rmsprop': moving average of squared gradients (stable learning rate)
    #optimizer = 'adam': first and second order moments with momentum
    
    #metrics = ['accuracy']: evaluates via accuracy 
    #metrics = ['mse', 'mae']: evaluates with error (use for regression)
    
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=SGD(),
    metrics=['accuracy'])

#EVALUATE
model.fit(
    X_train,
    y_train,
    epochs=100,                             #iterations through the training set 
    verbose=0,                              #number of epoch prints
    validation_data = [X_test, y_test])     #validation data

loss, test_acc = model.evaluate(X_test, y_test, verbose=0)      #loss and accuracy
y_pred_probs = model.predict(X, verbose=0)                      #predict class probabilities
predicted_class_indices = np.argmax(y_pred_probs, axis=1)       #convert probabilities into labels

'''
M4 Lab-2: LSTM for IMDB sentiment analysis
This lab is based on ISLP Chapter 10.
The objective is to correctly classify reviews into positive vs. negative ones.
Please first review the IMDB dataset documentaion on Keras.

Step 0: Load Libraries
Load the libraries you need in this lab
'''

# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings

import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# FREEZE CODE END

'''
Step 1: Load & Prep Dataframes

[Hyperparameter Tuning Zone]
It's good practice to keep important hyperparameters all in one place
So we can tune them easily later
The default value of 0 is provided here
You need to change these hyperparameter values here based on coding instructives given below
There is nothing to print in this section

CAUTION: Hyperparameter values are set low here to ensure the lab runs properly on Codio.
         On Colab you should feel free to try different values to improve model performance.
'''

max_features = 0
maxlen = 0
embedding_size = 0
batch_size = 0
num_epochs = 0
lstm_units = 0
dropout = 0

# WRITE YOUR CODE HERE

'''
Load the IMDB dataset from Keras using the dataset module
Keep only 5000 most frequent words
Reminder: Set such hyperparameter value in the [Hyperparameter Tuning Zone] above
Split the dataset into train vs. test automatically while reading in the dataset
We will refer to each record (row) as a sequence
**Q4-2-0 Print "xxx sequences in x_train" - Replace xxx with the size of the train sequences.
**Q4-2-1 Print the first 12 elements in the first sequence in a new line
**Q4-2-2 Print "First y value is xxx" - Replace xxx with the first element in y test in a new line.
'''
# WRITE YOUR CODE HERE
from keras.datasets import imdb 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

print(f"{len(x_train)} sequences in x_train")
print(x_train[0][:12])
print(f"First y value is {y_test[0]}")

'''
Step 2: Prep train vs. test sets
Pad the sequences with zeros AFTER each sequence in both train and test sets
So the sequences are of the same lengths of 50
**Q4-2-3 Print "x_train shape: xxx" - Replace xxx with the shape of x train set
**Q4-2-4 Print "x_test shape: xxx" - Replace xxx with the shape of x test set
**Q4-2-5 Print "y_train shape: xxx" - Replace xxx with the shape of y train set
**Q4-2-6 Print "y_test shape: xxx" - Replace xxx with the shape of y test 
'''
# WRITE YOUR CODE HERE
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train_padded = pad_sequences(x_train, maxlen=50, padding='post')
x_test_padded = pad_sequences(x_test, maxlen=50, padding='post')
print(f"x_train shape: {x_train_padded.shape}")
print(f"x_test shape: {x_test_padded.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

'''
Step 3: Train model
Build a Sequential() model with 
1. A dense embedding layer with the appropriate input dimension, and 36 as the output dimension of embedding
2. An LSTM layer with the tanh activation function, 26 output units, and a 10% dropout rate
3. A dense output layer with the sigmoid activation function and an appropriate output dimension
Do NOT name any of the layers
**Q4-2-7 Print your model's summary
'''
# WRITE YOUR CODE HERE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=36))                    # 1. Embedding layer
model.add(LSTM(units=26, activation='tanh', dropout=0.1))              # 2. LSTM layer
model.add(Dense(1, activation='sigmoid'))                              # 3. Output layer

model.summary()

'''
CAUTION: This step is the most computationally intensive
         Feel free to comment out this code segment when you're running Check It buttons for previous steps
Compile your model with 
- binary_crossentropy as the loss function
- adam as the optimizer
- accuracy as the metrics

Train your model with 
- the train and test sets you created above
- batch size of 256
- 5 epochs (to minimize demand for computing resources)
Turn off printing epoch outputsm so it's easier for you to inspect the output

Step 4: Evaluate model performance
Use the history property of the model fit output
**Q4-2-8 Print** training accuracy
**Q4-2-9 Print** validation accuracy    
'''

# WRITE YOUR CODE HERE
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(
    x_train_padded, y_train,
    validation_data=(x_test_padded, y_test),
    batch_size=256,
    epochs=5,
    verbose=0)

print(history.history['accuracy'])

print(history.history['val_accuracy'])'''
M4 Lab-2: LSTM for IMDB sentiment analysis
This lab is based on ISLP Chapter 10.
The objective is to correctly classify reviews into positive vs. negative ones.
Please first review the IMDB dataset documentaion on Keras.

Step 0: Load Libraries
Load the libraries you need in this lab
'''

# FREEZE CODE BEGIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings

import random
import numpy as np
import tensorflow as tf

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)
# FREEZE CODE END

'''
Step 1: Load & Prep Dataframes

[Hyperparameter Tuning Zone]
It's good practice to keep important hyperparameters all in one place
So we can tune them easily later
The default value of 0 is provided here
You need to change these hyperparameter values here based on coding instructives given below
There is nothing to print in this section

CAUTION: Hyperparameter values are set low here to ensure the lab runs properly on Codio.
         On Colab you should feel free to try different values to improve model performance.
'''

max_features = 0
maxlen = 0
embedding_size = 0
batch_size = 0
num_epochs = 0
lstm_units = 0
dropout = 0

# WRITE YOUR CODE HERE

'''
Load the IMDB dataset from Keras using the dataset module
Keep only 5000 most frequent words
Reminder: Set such hyperparameter value in the [Hyperparameter Tuning Zone] above
Split the dataset into train vs. test automatically while reading in the dataset
We will refer to each record (row) as a sequence
**Q4-2-0 Print "xxx sequences in x_train" - Replace xxx with the size of the train sequences.
**Q4-2-1 Print the first 12 elements in the first sequence in a new line
**Q4-2-2 Print "First y value is xxx" - Replace xxx with the first element in y test in a new line.
'''
# WRITE YOUR CODE HERE
from keras.datasets import imdb 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

print(f"{len(x_train)} sequences in x_train")
print(x_train[0][:12])
print(f"First y value is {y_test[0]}")

'''
Step 2: Prep train vs. test sets
Pad the sequences with zeros AFTER each sequence in both train and test sets
So the sequences are of the same lengths of 50
**Q4-2-3 Print "x_train shape: xxx" - Replace xxx with the shape of x train set
**Q4-2-4 Print "x_test shape: xxx" - Replace xxx with the shape of x test set
**Q4-2-5 Print "y_train shape: xxx" - Replace xxx with the shape of y train set
**Q4-2-6 Print "y_test shape: xxx" - Replace xxx with the shape of y test 
'''
# WRITE YOUR CODE HERE
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train_padded = pad_sequences(x_train, maxlen=50, padding='post')
x_test_padded = pad_sequences(x_test, maxlen=50, padding='post')
print(f"x_train shape: {x_train_padded.shape}")
print(f"x_test shape: {x_test_padded.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

'''
Step 3: Train model
Build a Sequential() model with 
1. A dense embedding layer with the appropriate input dimension, and 36 as the output dimension of embedding
2. An LSTM layer with the tanh activation function, 26 output units, and a 10% dropout rate
3. A dense output layer with the sigmoid activation function and an appropriate output dimension
Do NOT name any of the layers
**Q4-2-7 Print your model's summary
'''
# WRITE YOUR CODE HERE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=36))                    # 1. Embedding layer
model.add(LSTM(units=26, activation='tanh', dropout=0.1))              # 2. LSTM layer
model.add(Dense(1, activation='sigmoid'))                              # 3. Output layer

model.summary()

'''
CAUTION: This step is the most computationally intensive
         Feel free to comment out this code segment when you're running Check It buttons for previous steps
Compile your model with 
- binary_crossentropy as the loss function
- adam as the optimizer
- accuracy as the metrics

Train your model with 
- the train and test sets you created above
- batch size of 256
- 5 epochs (to minimize demand for computing resources)
Turn off printing epoch outputsm so it's easier for you to inspect the output

Step 4: Evaluate model performance
Use the history property of the model fit output
**Q4-2-8 Print** training accuracy
**Q4-2-9 Print** validation accuracy    
'''

# WRITE YOUR CODE HERE
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(
    x_train_padded, y_train,
    validation_data=(x_test_padded, y_test),
    batch_size=256,
    epochs=5,
    verbose=0)

print(history.history['accuracy'])

print(history.history['val_accuracy'])