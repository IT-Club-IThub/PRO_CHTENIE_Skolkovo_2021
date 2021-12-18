from LSTM_NN import LSTM_NN
# from BOW_NN import BOW_NN
# from Embedding_NN import Embedding_NN
# from RNN_NN import RNN_NN
import numpy as np
import os
from datetime import datetime

def main():
    dataset_path = "NEWNEWdataset_numWords_49209_xLen_300_step_300_.npz"
    print("--------------" + dataset_path.upper() + "--------------")
    dataset = '../datasets/' + dataset_path
    debug = True
    model = LSTM_NN(dataset, debug=debug, batch_size=128, step=85)
    model.compile()
    # model.model.load_weights('title2')
    while(True):
        model.fit(1)
        model.check()
        model.model.save_weights('alltrue')
        
if __name__ == "__main__":
    main()
