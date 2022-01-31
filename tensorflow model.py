import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from tensorflow.random import set_seed
import matplotlib.pyplot as plt

np.random.seed(37)
set_seed(37)

data = np.load('processed.npy')
target =  pd.read_csv('train.csv', usecols = ['SalePrice']).to_numpy()
train, test, train_target, test_target = train_test_split(data, target, shuffle = False)

optimizer = Adam(learning_rate = 0.01)
model = Sequential()

model.add(Dense(units = 128,input_dim=train.shape[1], activation = 'relu'))
model.add(Dense(units = 16, activation = 'relu'))
model.add(Dense(units = 1, activation = 'linear'))

model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse', 'mae'])

history = model.fit(train, train_target, epochs = 100,validation_split=0.2, verbose=False, use_multiprocessing=True)
model.save_weights('models/tensorflow_model.h5')
history_df = pd.DataFrame(history.history)
plt.plot(history_df['loss'], label = 'loss')
plt.plot(history_df['val_loss'], label = 'val_loss')
print("val_mae = ", list(history_df["val_mae"])[-1])
plt.xlabel("epochs")
plt.legend()
plt.show()

model.load_weights('models/tensorflow_model.h5')
y_pred = model.predict(test).flatten()
mn = 0
mx = 800000
plt.figure(figsize=(7,7))
a = plt.axes(aspect='equal')
plt.scatter(test_target, y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Actual vs predicted values')
plt.xlim([mn, mx])
plt.ylim([mn, mx])
plt.grid()
plt.plot([mn, mx], [mn, mx])
plt.show()

test_target = pd.DataFrame(test_target, columns = ['true sale'])
y_pred = model.predict(test).flatten()
sub = pd.concat([pd.DataFrame(y_pred, columns = ['prediction']), test_target], axis = 1)
sub.to_csv('submission.csv', index = False)
result = pd.read_csv('submission.csv')