from keras.models import load_model
from keras.models import Model
from PIL import Image, ImageEnhance
import numpy as np

validate_data = np.stack([(np.array(Image.open("./data/8dataset/8-" + str(index) +".jpg")))/255.0 for index in range(1, 11, 1)])
model = load_model("./data/cnn_model.h5")
prediction = model.predict(validate_data)
resultlist = ["" for _ in range(10)]
for predict in prediction:
    for index in range(10):
        resultlist[index] += str(np.argmax(predict[index]))
for result in resultlist:
    print(result)
