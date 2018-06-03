from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import csv

outputcsv = open('./data/6_real_train_set/captcha_train.csv', 'w', encoding = 'utf8', newline = '') # 輸出csv
inputcsv = open('./data/manual_label/captcha_vali.csv', 'r', encoding = 'utf8')
data = [np.array(Image.open('./data/manual_label/' + row[0] + ".jpg")) for row in csv.reader(inputcsv) if len(row[1]) == 6] # 只讀答案是6位的
inputcsv = open('./data/manual_label/captcha_vali.csv', 'r', encoding = 'utf8')
oldanswer = [row[1] for row in csv.reader(inputcsv) if len(row[1]) == 6] # 只讀答案是6位的
answer = []
datagen = ImageDataGenerator(rotation_range=5,shear_range=0.2,zoom_range=0.05,fill_mode='nearest')
index, augmentindex, oldanswerindex = 0, 0, 0
for img in data:
    for batch in datagen.flow(np.asarray([img]), batch_size=1):
        index += 1
        augmentindex += 1
        batch = batch.reshape((60,200,3))
        Image.fromarray(np.uint8(batch)).convert("RGB").save("./data/6_real_train_set/" + str(index) + ".jpg", "JPEG")
        answer.append((str(index), oldanswer[oldanswerindex]))
        if augmentindex >= 50: # 每張產生50個
            oldanswerindex += 1
            augmentindex = 0
            break
csv.writer(outputcsv).writerows(answer)
