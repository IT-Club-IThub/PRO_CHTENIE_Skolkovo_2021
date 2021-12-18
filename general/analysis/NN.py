from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, BatchNormalization, Embedding, LSTM
import numpy as np

class NN:
  def __init__(self, dataset_path: str, debug: bool,batch_size:int, step: int, epochs:int=15):
    self.dataset_path = dataset_path
    self.debug = debug
    self.batch_size = batch_size
    self.step = step

    dataset = np.load(dataset_path, allow_pickle=True)
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
    self.xLen = int(filename.split('_')[4])

    

  

  def fit(self, epochs:int=15):
    self.historyLSTM = self.model.fit(self.xTrain, 
                                self.yTrain, 
                                epochs=epochs,
                                batch_size=self.batch_size,
                                validation_data=(self.xVal, self.yVal),
                                use_multiprocessing=True,
                                verbose=int(self.debug))

  def _get_max_count_value(self, arr):
    db = {}
    for element in arr:
      if (element in db.keys()):
        db[element] += 1
      else:
        db[element] = 1
    return max(db, key=db.get)


  def _get_answer_from_part(self, part):
      pred = self.model.predict(np.array([part]))
      print(pred)
      

  def check(self):
    yes = {0:0, 1:0}
    no = {0:0, 1:0}
    for index in range(len(self.xTest)):
      part = self.xTest[index]
      answer = self.yTest[index][0]

      local_answer = self._get_answer_from_part(part)


    # rightAnswer = [0,0]
    # totalAnswer = [0,0]
    # currPred = self.model.predict(self.xTest)
    # currOut = np.argmax(currPred, axis=1)
    # yOut = np.argmax(self.yTest, axis=1)
    # print(currOut)
    # print(yOut)
    # for i in range(len(yOut)):
    #   predictA = currOut[i]
    #   rightA   = yOut[i]
    #   totalAnswer[rightA] += 1
    #   if predictA == rightA:
    #     rightAnswer[rightA] += 1

    # # print(f"Точность распознавания текстов на {self.dataset_path}")
    # for i in range(2):
    #   print("{:12s}: {:3d} из {:3d} - {:3.2f}%".format(str(i), rightAnswer[i], totalAnswer[i], (rightAnswer[i]/totalAnswer[i]*100)))
    
# в xTest: полное сочинение
