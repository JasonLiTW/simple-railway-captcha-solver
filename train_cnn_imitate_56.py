from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import numpy as np
import csv


# Create CNN Model
print("Creating CNN model...")
in = Input((60, 200, 3))
out = in
out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(0.5)(out)
out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Flatten()(out)
out = Dropout(0.5)(out)
out = Dense(1, name='6digit', activation='sigmoid')(out)
model = Model(inputs=in, outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Reading training data...")
traincsv = open('./data/56_imitate_train_set/len_train.csv', 'r', encoding = 'utf8')
train_data = np.stack([np.array(Image.open("./data/56_imitate_train_set/" + row[0] + ".jpg"))/255.0 for row in csv.reader(traincsv)])
traincsv = open('./data/56_imitate_train_set/len_train.csv', 'r', encoding = 'utf8')
train_label = np.asarray([1 if row[1] == '6' else 0 for row in csv.reader(traincsv)])
print("Shape of train data:", train_data.shape)

print("Reading validation data...")
valicsv = open('./data/56_imitate_vali_set/len_vali.csv', 'r', encoding = 'utf8')
vali_data = np.stack([np.array(Image.open('./data/56_imitate_vali_set/' + row[0] + ".jpg"))/255.0 for row in csv.reader(valicsv)])
valicsv = open('./data/56_imitate_vali_set/len_vali.csv', 'r', encoding = 'utf8')
vali_label = np.asarray([1 if row[1] == '6' else 0 for row in csv.reader(valicsv)])
print("Shape of validation data:", vali_data.shape)

filepath="./data/model/imitate_56_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
callbacks_list = [checkpoint, earlystop, tensorBoard]
model.fit(train_data, train_label, batch_size=400, epochs=100, verbose=1, validation_data=(vali_data, vali_label), callbacks=callbacks_list)
