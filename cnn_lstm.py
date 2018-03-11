'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

batch_size = 50
num_classes = 4
epochs = 250

# input image dimensions
img_rows, img_cols = 128, 256

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

'''
import pickle as pk

train_pkl = pk.load(open("1800.pkl"))
test_pkl = pk.load(open("450.pkl"))

x_train = np.asarray(train_pkl[0]).reshape(1800,128,256,1)
print x_train.shape
y_train = keras.utils.to_categorical(np.asarray(train_pkl[1]), num_classes=4)
print y_train.shape

x_test = np.asarray(test_pkl[0]).reshape(450,128,256,1)
print x_train.shape
y_test = keras.utils.to_categorical(np.asarray(test_pkl[1]), num_classes=4)
print y_train.shape
'''


'''
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
'''

input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train += 80
x_test += 80
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

'''
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''


model = Sequential()

model.add(Conv2D(32, kernel_size=(12, 16),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(48, (8, 12), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (5, 7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
print(model.summary())
model.add(Reshape((640, 24), input_shape=(10, 24, 64)))
print(model.summary())
model.add(Bidirectional(LSTM(128,batch_size=batch_size,return_sequences=True,input_shape=(10,24,64))))
#model.add(Dropout(0.5))
print(model.summary())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
