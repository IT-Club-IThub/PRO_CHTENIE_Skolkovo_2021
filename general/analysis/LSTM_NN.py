from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, BatchNormalization, Embedding, LSTM
from NN import NN

class LSTM_NN(NN):
  def compile(self):
    self.model = Sequential()
    self.model.add(Embedding(self.numWords, self.step, input_length=self.xLen))
    self.model.add(SpatialDropout1D(0.3))
    self.model.add(BatchNormalization())
    self.model.add(LSTM(30, return_sequences=1))
    self.model.add(LSTM(15))
    self.model.add(Dense(2, activation='softmax'))
    self.model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    if(self.debug):
      self.model.summary()