from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
from PIL import Image
from keras.models import load_model, Model
import time
import random
IDNumber = "X123456789" # 填入你的身分證字號
model = None
model5 = load_model("./data/model/imitate_5_model.h5") # 辨識5碼的Model
model6 = load_model("./data/model/imitate_6_model.h5") # 辨識6碼的Model
model56 = load_model("./data/model/real_56_model.h5") # 辨識是5碼or6碼的Model
LETTERSTR = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
driver = webdriver.Chrome("./data/chromedriver.exe") # chromedriver 路徑
correct, wrong = 0, 0

for _ in range(1000):# 跑1000次
    driver.get('http://railway1.hinet.net/Foreign/TW/ecsearch.html')
    id_textbox = driver.find_element_by_id('person_id')
    id_textbox.send_keys(IDNumber)
    button = driver.find_element_by_css_selector('body > div.container > div.row.contents > div > form > div > div.col-xs-12 > button')
    button.click()
    driver.save_screenshot('tmp.png')
    location = driver.find_element_by_id('idRandomPic').location
    x, y = location['x'] + 5, location['y'] + 5
    img = Image.open('tmp.png')
    captcha = img.crop((x, y, x+200, y+60))
    captcha.convert("RGB").save('captcha.jpg', 'JPEG')
    # check is 5 or 6 digits
    p56 = model56.predict(np.stack([np.array(Image.open('captcha.jpg'))/255.0]))[0][0]
    if p56 > 0.5:
        model = model6
    else:
        model = model5
    prediction = model.predict(np.stack([np.array(Image.open('captcha.jpg'))/255.0]))
    answer = ""
    for predict in prediction:
        answer += LETTERSTR[np.argmax(predict[0])]
    captcha_textbox = driver.find_element_by_id('randInput')
    captcha_textbox.send_keys(answer)
    driver.find_element_by_id('sbutton').click()
    if "亂數號碼錯誤" in driver.page_source:
        wrong += 1
    else:
        correct += 1
    print("{:.4f}% (Correct{:d}-Wrong{:d})".format(correct/(correct+wrong)*100, correct, wrong))
    time.sleep(3)
