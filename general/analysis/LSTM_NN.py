from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, BatchNormalization, Embedding, LSTM
import numpy as np

class LSTM_NN:
  def __init__(self, dataset_path: str, debug: bool,batch_size:int, epochs:int=15):
    self.dataset_path = dataset_path
    self.debug = debug
    self.batch_size = batch_size

    dataset = np.load(dataset_path)
    self.xTrain = dataset['xTrain']
    self.yTrain = dataset['yTrain']
    self.xTest = dataset['xTest']
    self.yTest = dataset['yTest']
    self.xVal = dataset['xVal']
    self.yVal = dataset['yVal']
  
    if(self.debug):
      print(self.xTrain.shape)
      print(self.yTrain.shape)
      print(self.xTest.shape)
      print(self.yTest.shape)
      print(self.xVal.shape)
      print(self.yVal.shape)

    filename = dataset_path.split('/')[-1]
    self.numWords = int(filename.split('_')[2])
    self.xLen = int(filename.split('_')[5])
    # self.batch_size = 32
    self.step = int(filename.split('_')[7])
    self.units_LSTM = 10
    self.epochs = epochs

    

  def compile(self):
    self.model = Sequential()
    self.model.add(Embedding(self.numWords, self.step, input_length=self.xLen))
    self.model.add(SpatialDropout1D(0.2))
    self.model.add(BatchNormalization())
    self.model.add(LSTM(10))
    # self.modelLSTM.add(LSTM(5))
    # modelLSTM2.add(LSTM(5))

    self.model.add(Dense(2, activation='softmax'))
    self.model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    if(self.debug):
      self.model.summary()

  def fit(self):
    self.historyLSTM = self.model.fit(self.xTrain, 
                                self.yTrain, 
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                validation_data=(self.xVal, self.yVal),
                                use_multiprocessing=True,
                                verbose=int(self.debug))

  def check(self):
    rightAnswer = [0,0]
    totalAnswer = [0,0]
    currPred = self.model.predict(self.xTest)
    currOut = np.argmax(currPred, axis=1)
    yOut = np.argmax(self.yTest, axis=1)
    for i in range(len(yOut)):
      predictA = currOut[i]
      rightA   = yOut[i]
      totalAnswer[rightA] += 1
      if predictA == rightA:
        rightAnswer[rightA] += 1

    # print(f"Точность распознавания текстов на {self.dataset_path}")
    for i in range(2):
      print("{:12s}: {:3d} из {:3d} - {:3.2f}%".format(str(i), rightAnswer[i], totalAnswer[i], (rightAnswer[i]/totalAnswer[i]*100)))