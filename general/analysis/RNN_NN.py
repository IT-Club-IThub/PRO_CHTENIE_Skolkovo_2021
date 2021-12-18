from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, BatchNormalization, Embedding, SimpleRNN
from NN import NN

class RNN_NN(NN): 
  def compile(self):
    self.model = Sequential()
    self.model.add(Embedding(self.numWords, 5, input_length=self.xLen))
    self.model.add(SpatialDropout1D(0.2))
    self.model.add(BatchNormalization())
    self.model.add(SimpleRNN(16, dropout=0.2, recurrent_dropout=0.2, activation="relu"))
    self.model.add(Dense(2, activation='softmax'))
    self.model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    if(self.debug):
      self.model.summary()