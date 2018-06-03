from keras.models import load_model
from keras.models import Model
from keras import backend as K
from PIL import Image
import numpy as np
import os
import csv
LETTERSTR = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(34)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist


print("Loading test data...")
testcsv = open('./data/manual_label/captcha_test.csv', 'r', encoding = 'utf8')
test_data = np.stack([np.array(Image.open("./data/manual_label/" + row[0] + ".jpg"))/255.0 for row in csv.reader(testcsv)])
testcsv = open('./data/manual_label/captcha_test.csv', 'r', encoding = 'utf8')
test_label = [row[1] for row in csv.reader(testcsv)]
print("Loading model...")
K.clear_session()
model = None
model5 = load_model("./data/model/imitate_5_model.h5")
model6 = load_model("./data/model/imitate_6_model.h5")
model56 = load_model("./data/model/real_56_model.h5")
print("Predicting...")
prediction56 = [6 if arr[0] > 0.5 else 5 for arr in model56.predict(test_data)] # 5/6碼分類
prediction5 = model5.predict(test_data) # 5碼
prediction6 = model6.predict(test_data) # 6碼

# 以下計算各個模型各個字元辨識率等等，有點亂，以後有空再整理
total, total5, total6 = len(prediction56), 0, 0
correct5, correct6, correct56, correct = 0, 0, 0, 0
correct5digit, correct6digit = [0 for _ in range(5)], [0 for _ in range(6)]
totalalpha, correctalpha = len([1 for ans in test_label for char in ans if char.isalpha()]), 0
for i in range(total):
    checkcorrect = True
    if prediction56[i] == len(test_label[i]):
        correct56 += 1
    else:
        checkcorrect = False
    if prediction56[i] == 5:
        total5 += 1
        allequal = True
        for char in range(5):
            if LETTERSTR[np.argmax(prediction5[char][i])] == test_label[i][char]:
                correct5digit[char] += 1
                correctalpha += 1 if LETTERSTR[np.argmax(prediction5[char][i])].isalpha() else 0
            else:
                allequal = False
        if allequal:
            correct5 += 1
        else:
            checkcorrect = False
    else:
        total6 += 1
        allequal = True
        for char in range(6):
            if LETTERSTR[np.argmax(prediction6[char][i])] == test_label[i][char]:
                correct6digit[char] += 1
                correctalpha += 1 if LETTERSTR[np.argmax(prediction6[char][i])].isalpha() else 0
            else:
                allequal = False
        if allequal:
            correct6 += 1
        else:
            checkcorrect = False
    if checkcorrect:
        correct += 1

print("5 or 6 model acc:{:.4f}%".format(correct56/total*100)) # 5/6模型acc
print("---------------------------")
print("5digits model acc:{:.4f}%".format(correct5/total5*100)) # 5模型acc
for i in range(5):
    print("digit{:d} acc:{:.4f}%".format(i+1, correct5digit[i]/total5*100)) # 5模型各字元acc
print("---------------------------")
print("6digits model acc:{:.4f}%".format(correct6/total6*100)) # 6模型acc
for i in range(6):
    print("digit{:d} acc:{:.4f}%".format(i+1, correct6digit[i]/total6*100)) # 6模型各字元acc
print("---------------------------")
print("alpha acc:{:.4f}%".format(correctalpha/totalalpha*100)) # 整體英文字acc
