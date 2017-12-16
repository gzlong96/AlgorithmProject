from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import numpy as np
import data_reader
import random

np.random.seed(1234567)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
model.add(Dense(64, activation='relu', input_dim=data_reader.fft_n*4))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=0.01)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

all_seq, all_label = data_reader.read_all()
y = all_label[0]
x = all_seq[0]

acc_history=[]

for i in range(10): #10 rounds
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    selects = []
    for j in range(5):
        num = random.randint(0,35)
        while num in selects:
            num = random.randint(0, 35)
        selects.append(num)
    for j in range(36):
        if j in selects:
            x_test.append(x[j])
            y_test.append(y[j])
        else:
            x_train.append(x[j])
            y_train.append(y[j])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print x_train.shape
    model.fit(x_train, y_train, epochs=250, batch_size=16,shuffle=True)
    score = model.evaluate(x_test, y_test, batch_size=5)
    acc_history.append(score[1])
    print "test "+str(i+1)+str(score)

score = model.evaluate(x, y, batch_size=36)
print "final test " + str(score)
print sum(acc_history)/10
# model.save("/root/merge_model")
