# simple-railway-captcha-solver
![image](./readme_img/1.jpeg)

本專案利用簡單的Convolutional Neural Network來實作辨識台鐵訂票網站的驗證碼(如上圖)，訓練集的部分以模仿驗證碼樣式的方式來產生、另外驗證集的部分則自台鐵訂票網站擷取，再以手動方式標記約1000筆。

目前驗證集對於6碼型態的驗證碼的單碼辨識率達到98.84%，整體辨識成功率達到91.13%。底下有詳盡的說明。

(僅供學術研究用途，切勿違法使用於訂票或其他用途，以免觸犯相關法規。)

|Name|Description|
|----|----|
|captcha_gen.py|模仿驗證碼樣式建立訓練集|
|train_cnn.py  |建立並訓練CNN|
|demo_solver.py|Demo載入模型並辨識驗證碼|

## Dependecies
|Name|Version|
|----|----|
|tensorflow|1.4.0|
|tensorflow-gpu|1.4.0|
|tensorflow-tensorboard|0.4.0rc3|
|Keras|2.1.2|
|h5py|2.7.1|
|Pillow|4.3.0|
|numpy|1.13.3|

## Training data?
要建立一個辨識驗證碼的CNN模型其實並非難事，難的是要如何取得標記好的訓練集呢?

![image](./readme_img/2.jpeg)![image](./readme_img/3.jpeg)![image](./readme_img/4.jpeg)

或許你可以寫一支爬蟲程式，擷取個5萬張驗證碼，再自己手動標記答案上去，但這樣有點太費時了，或許我們可以試著模仿產生一些驗證碼看看。
不過當然，我們產生的訓練集必須非常接近真實的驗證碼，否則最後訓練完可能用在真實的驗證碼上效果會非常的差。

## Generate training data

首先我們要先觀察驗證碼，你可以寫一支爬蟲程式去擷取一兩百張驗證碼回來細細比對。我們不難發現台鐵的驗證碼不外乎由兩個主要元素組成：
- 5 ~ 6碼的數字，大小似乎不一致，而且都有經過旋轉，另外顏色是浮動的。
- 背景是浮動的顏色，另外還有不少干擾的線條，看起來應該是矩形，由黑線和白線組成，且有部分會蓋到數字上面。

進一步研究會發現:
- 數字的旋轉角度約在-55 ~ 55度間，大小約25 ~ 27pt。
- 字型的部分，仔細觀察會發現同一個字會有兩種不一樣的樣式，推測是有兩種字型隨機更替，其中一個很明顯是Courier New-Bold，另一個比對一下也不難發現即是Times New Roman-Bold。
- 背景和字型顏色的部分，可以用一些色彩均值化的手法快速的從數百張的驗證碼中得出每一張的背景及數字的顏色，進而我們就能算出顏色的範圍。這部分可以用opencv的k-means來實作，這邊就不再贅述。
背景的R/G/B範圍約是在180 ~ 250間，文字的部分則是10 ~ 140間。
- 干擾的線條是矩形，有左、上是黑線條且右、下是白線條和倒過來，共兩種樣式(也可以當作是旋轉180度)，平均大約會出現30 ~ 32個隨機分布在圖中，長寬都大約落在5 ~ 21px間。
另外，大約有4成的機會白線會蓋在數字上，黑線蓋在文字上的機率則更低。

有了這些觀察，只差一點點就可以產生訓練集了，我們現在來觀察數字都落在圖片上的甚麼位置上:

![image](./readme_img/5.PNG)![image](./readme_img/6.PNG)![image](./readme_img/7.PNG)

從這幾張圖中不難看出文字並非規則地分布在圖片上，我們可以猜測文字是旋轉後被隨機左移或右移了，甚至還會有重疊的情況，所以沒辦法用切割的方式一次處理一個文字。

以上就是我們簡單觀察到的驗證碼規則，訓練集產生的部分實作在captcha_gen.py中，雖然寫得有點雜亂，不過沒甚麼特別的地方，就是照著上面的規則產生，可以試著以自己的方式實作看看。

![image](./readme_img/8.jpg)

最後會輸出5萬張驗證碼圖片和1個標記答案的csv檔。

![image](./readme_img/9.PNG)![image](./readme_img/10.PNG)


## Building Convolution Neural Network
來建立一個簡單結構的CNN吧!

輸入是60*200的圖片，共有3個channel(R/G/B)，所以是shape會是(60, 200, 3)。

中間透過數層由ReLU函數激發的Convolution Layer擷取特徵，並以2x2的Max pooling layer採樣減少計算量，最後經過一個Dropout Layer隨機捨棄一些神經元(避免overfitting)和Flatten Layer來把資料降到1維，輸出到全連接層：6個10神經元的Softmax regression分類器。

```python
tensor_in = Input((60, 200, 3))
out = tensor_in
out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Flatten()(out)
out = Dropout(0.5)(out)
out = [Dense(10, name='digit1', activation='softmax')(out),\
    Dense(10, name='digit2', activation='softmax')(out),\
    Dense(10, name='digit3', activation='softmax')(out),\
    Dense(10, name='digit4', activation='softmax')(out),\
    Dense(10, name='digit5', activation='softmax')(out),\
    Dense(10, name='digit6', activation='softmax')(out)]
model = Model(inputs=tensor_in, outputs=out)
```

完成後要來compile模型，這邊loss使用categorical_crossentropy、optimizer*使用Adamax，而metrics理所當然是accuracy了。
```python
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
```

*關於optimizer的選擇，可以參考這兩篇，寫得不錯：
1. An overview of gradient descent optimization algorithms -  http://ruder.io/optimizing-gradient-descent/index.html
2. SGD，Adagrad，Adadelta，Adam等优化方法总结和比较 - http://ycszen.github.io/2016/08/24/SGD%EF%BC%8CAdagrad%EF%BC%8CAdadelta%EF%BC%8CAdam%E7%AD%89%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95%E6%80%BB%E7%BB%93%E5%92%8C%E6%AF%94%E8%BE%83/
**

--

最後來看看model的summary輸出長甚麼樣子：
```python
model.summary()

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
======================================================
input_1 (InputLayer)            (None, 60, 200, 3)   0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 60, 200, 32)  896         input_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 58, 198, 32)  9248        conv2d_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 29, 99, 32)   0           conv2d_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 29, 99, 64)   18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 27, 97, 64)   36928       conv2d_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 13, 48, 64)   0           conv2d_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 13, 48, 128)  73856       max_pooling2d_2[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 11, 46, 128)  147584      conv2d_5[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 5, 23, 128)   0           conv2d_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 3, 21, 256)   295168      max_pooling2d_3[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 1, 10, 256)   0           conv2d_7[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 2560)         0           max_pooling2d_4[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 2560)         0           flatten_1[0][0]
__________________________________________________________________________________________________
digit1 (Dense)                  (None, 10)           25610       dropout_1[0][0]
__________________________________________________________________________________________________
digit2 (Dense)                  (None, 10)           25610       dropout_1[0][0]
__________________________________________________________________________________________________
digit3 (Dense)                  (None, 10)           25610       dropout_1[0][0]
__________________________________________________________________________________________________
digit4 (Dense)                  (None, 10)           25610       dropout_1[0][0]
__________________________________________________________________________________________________
digit5 (Dense)                  (None, 10)           25610       dropout_1[0][0]
__________________________________________________________________________________________________
digit6 (Dense)                  (None, 10)           25610       dropout_1[0][0]
=======================================================
Total params: 735,836
Trainable params: 735,836
Non-trainable params: 0
```

架構以圖片呈現的話:

![image](./readme_img/11.PNG)

## Load training data
在訓練之前我們要先將資料載入到記憶體中，前面產生訓練集的時候，我們是將驗證碼存成一張張編號好的圖片，並用csv檔記錄下了答案。

首先我們先處理X的部分，也就是特徵值，這邊就是指我們的圖片。
而要輸入進CNN的資料必須是numpy array的形式，所以我們用Pillow來讀取圖片並轉為numpy格式：

```python
for index in range(1, 50001, 1)
    image = Image.open("./data/train_set/" + str(index) + ".jpg")
    nparr = np.array(image)
    nparr = nparr / 255.0
```

這時我們下```nparr.shape```，可以看到矩陣的大小是```(60, 200, 3)```，跟前面模型設計的Input是相同的。

而我們計劃使用50000張圖片來訓練，所以最後輸入給CNN的矩陣大小會是```(50000, 60, 200, 3)```，這部分只要利用stack就可以把它們合併，整理成下面:

```python
train_data = np.stack([np.array(Image.open("./data/train_set/" + str(index) + ".jpg"))/255.0 for index in range(1, 50001, 1)])
```

最後train_data的shape就會是```(50000, 60, 200, 3)```。

接下來Y則是訓練集的標記，也就是我們訓練集的答案。

因為我們的模型是多輸出的結構(6組softmax函數分類器)，所以Y要是一個含有6個numpy array的list，大概像是這樣：
```
[[第一張第1個數字,...,最後一張第1個數字], [第一張第2個數字,...,最後一張第2個數字], [...], [...], [...], [...]]
```
而其中每個數字都是以one-hot encoding表示，例如2就是```[0, 0, 1, 0, ....,0]```

```python
traincsv = open('./data/train_set/train.csv', 'r', encoding = 'utf8')
read_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
train_label = [[] for _ in range(6)]
for arr in read_label:
    for index in range(6):
        train_label[index].append(arr[index])
train_label = [arr for arr in np.asarray(train_label)]
```

## Validation set
因為擔心模仿產生的訓練集沒辦法有效的使用在真實的驗證碼上，因此驗證集的部分我是用的是自行手動標記的真實驗證碼，這部分資料載入和處理方式跟上面相同，data放在vali_data，label放在vali_label。

## Callback
在這邊要用到三個callback:

1.ModelCheckPoint

```python
checkpoint = ModelCheckpoint(filepath, monitor='val_digit6_acc', verbose=1, save_best_only=True, mode='max')
```

用於儲存最佳辨識率的模型，每次epoch完會檢查一次，如果比先前最佳的acc高，就會儲存model到filepath。

因為在多輸出模型中沒有像是各輸出平均的acc這種東西，觀察前幾epoch後發現val_digit6_acc上升最慢，因此用它當作checkpoint的monitor。
(如果要自定義monitor可以自己寫callback，這部分留到未來有空再來實作。)

2.Earlystopping

```python
earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
```

這邊的monitor設為val_loss，patience設為2，也就是在驗證集的loss連續2次不再下降時，就會提早結束訓練。

3.TensorBoard

```python
tensorBoard = TensorBoard(log_dir = "./logs", histogram_freq = 1)
```

TensorBoard可以讓我們更方便的以圖形化界面檢視訓練結果，要檢視時可以輸入tensorboard --logdir=logs來啟動。

最後把他們三個放到list中即可。
```python
callbacks_list = [tensorBoard, earlystop, checkpoint]
```

## Train model
至此為止我們已經把所有需要的資料都準備好了，現在只需要一台好電腦就可以開始訓練了，建議使用GPU來訓練，不然要很久，真的很久....。

若在訓練時出現Resource exhausted的錯誤，可以考慮調低一些參數(如batch_size)。

```python
model.fit(train_data, train_label, batch_size=400, epochs=50, verbose=2, validation_data=(vali_data, vali_label), callbacks=callbacks_list)
```

## Result
模型在訓練15 epochs後達到EarlyStopping的條件停止。

驗證集的單碼辨識率達到平均98.84%，一次辨識成功率(即一次6碼都辨識正確)達到約91%。

![image](./readme_img/12.PNG)

## Problem & Todo
1. 目前只能固定辨識6碼的驗證碼，尚無法辨識5碼的。可能可以透過將5碼的標籤加入空白字元，或是後端再接一RNN的方式來實作。
2. 新增英文版的README。
3. 重寫captcha_gen.py，有點亂。

## Reference
1. An overview of gradient descent optimization algorithms -  http://ruder.io/optimizing-gradient-descent/index.html
2. SGD，Adagrad，Adadelta，Adam等优化方法总结和比较 - http://ycszen.github.io/2016/08/24/SGD%EF%BC%8CAdagrad%EF%BC%8CAdadelta%EF%BC%8CAdam%E7%AD%89%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95%E6%80%BB%E7%BB%93%E5%92%8C%E6%AF%94%E8%BE%83/
3. Going Deeper with Convolutions - http://arxiv.org/abs/1409.4842
