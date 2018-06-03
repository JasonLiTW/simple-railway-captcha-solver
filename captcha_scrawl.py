import shutil
import requests
import time
SAVEPATH = "./data/manual_label/"
url = 'http://railway1.hinet.net/ImageOut.jsp'
for i in range(1, 3000):
    response = requests.get(url, stream=True)
    with open(SAVEPATH + str(i) + '.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    time.sleep(0.5)
