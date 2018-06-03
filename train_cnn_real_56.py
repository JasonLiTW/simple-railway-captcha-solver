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
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

print("Reading training data...")
train_label = np.asarray([0 for _ in range(40000)])
train_data = [np.array(Image.open("./data/5_real_train_set/" + str(i) + ".jpg"))/255.0 for i in np.random.choice(range(1,60001), size=20000, replace=False)]
train_data = np.concatenate((train_data, [np.array(Image.open("./data/6_real_train_set/" + str(i) + ".jpg"))/255.0 for i in np.random.choice(range(1,60001), size=20000, replace=False)]))
train_data = np.stack(train_data)
train_label[:20000] = 0
train_label[20000:] = 1
print("Shape of train data:", train_data.shape)

print("Reading validation data...")
vali_label = np.asarray([0 for _ in range(10000)])
vali_data = [np.array(Image.open("./data/5_real_train_set/" + str(i) + ".jpg"))/255.0 for i in np.random.choice(range(60001,75001), size=5000, replace=False)]
vali_data = np.concatenate((vali_data, [np.array(Image.open("./data/6_real_train_set/" + str(i) + ".jpg"))/255.0 for i in np.random.choice(range(60001,75001), size=5000, replace=False)]))
vali_data = np.stack(vali_data)
vali_label[:5000] = 0
vali_label[5000:] = 1
print("Shape of validation data:", vali_data.shape)

filepath="./data/model/real_56_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq=1)
callbacks_list = [checkpoint, earlystop, tensorBoard]
model.fit(train_data, train_label, batch_size=400, epochs=100, verbose=2, validation_data=(vali_data, vali_label), callbacks=callbacks_list)
