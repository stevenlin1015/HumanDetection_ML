## 前言
以往長時間接觸行動裝置開發，而這學期修了系上的機器學習導論課程，也是我第一次接觸Machine Learning。
除了課堂理論的吸收之外，老師也以期末專題的方式讓自己更加了解實務面應用與操作。
雖然是第一次使用Keras（語言搭配Python），不過意外的是還蠻好上手的！
我想Keras對於第一次入門機器學習的人，算是相對親民的套件吧😀

****
## 概念與構想
學期初 老師原本要我們利用Nvidia Digits做影像辨識，後來也開放在CNN(Convolutional Neural Network, 卷積神經網路)一類，自行撰寫程式碼來做模型訓練。
後來我選擇後者，也因此有今天這個repo出現。
由於資料集要自己蒐集，剛好我本來就有在攝影，所以硬碟裡都是以前滿滿留下來的照片（資源Get!）
經過一番篩選與考量到資料數量問題，最後決定以每張照片中「有人」或是「沒有人」當做分類的結果。

****
## Overview
* 訓練資料與測試資料，包含：
    * 資料集的建立
    * 標籤方法
* 實作平台
* 硬體環境
* 辨識準確度
* 檢討與討論，包含：
    * 雜訊問題
    * 資料集數量
    * 心得
![](https://github.com/stevenlin1015/HumanDetection_ML/blob/master/ppt_export/投影片02.jpeg "Overview")

****
## 圖片蒐集完後的正規化
![](https://github.com/stevenlin1015/HumanDetection_ML/blob/master/ppt_export/投影片06.jpeg "圖片蒐集完後的正規化")

****
## 訓練資料與測試資料
![](https://github.com/stevenlin1015/HumanDetection_ML/blob/master/ppt_export/投影片07.jpeg "訓練資料與測試資料")
![](https://github.com/stevenlin1015/HumanDetection_ML/blob/master/ppt_export/投影片08.jpeg "訓練資料與測試資料")

****
## 實作平台
![](https://github.com/stevenlin1015/HumanDetection_ML/blob/master/ppt_export/投影片09.jpeg "實作平台")

****
## 硬體環境
![](https://github.com/stevenlin1015/HumanDetection_ML/blob/master/ppt_export/投影片10.jpeg "硬體環境")

****
## 辨識準確度
![](https://github.com/stevenlin1015/HumanDetection_ML/blob/master/ppt_export/投影片11.jpeg "辨識準確度")
![](https://github.com/stevenlin1015/HumanDetection_ML/blob/master/ppt_export/投影片12.jpeg "辨識準確度")
