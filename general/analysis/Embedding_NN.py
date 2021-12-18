from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, BatchNormalization, Embedding, Flatten
from NN import NN

class Embedding_NN(NN):
  def compile(self):
    self.model = Sequential()
    self.model.add(Embedding(self.numWords, 10, input_length=self.xLen))
    self.model.add(SpatialDropout1D(0.20))
    self.model.add(Flatten())
    self.model.add(BatchNormalization())
    self.model.add(Dense(2, activation='softmax'))
  
    self.model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    if(self.debug):
      self.model.summary()