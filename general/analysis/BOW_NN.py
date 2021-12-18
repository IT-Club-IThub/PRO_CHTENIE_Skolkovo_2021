from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from NN import NN
from tensorflow.python.keras.layers.core import Dropout

class BOW_NN(NN):
  def compile(self):
    self.model = Sequential()
    self.model.add(Dense(512, input_dim=self.numWords, activation="relu"))
    self.model.add(Dropout(0.25))
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