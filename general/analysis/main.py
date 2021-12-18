from LSTM_NN import LSTM_NN
from BOW_NN import BOW_NN
from Embedding_NN import Embedding_NN
from RNN_NN import RNN_NN
import os
from datetime import datetime

def main():
    dataset_path = "dataset_numWords_48131_xLen_300_step_(150, 35)_.npz"
    print("--------------" + dataset_path.upper() + "--------------")
    dataset = '../datasets/' + dataset_path
    debug = True
    step = 150
    model = LSTM_NN(dataset, debug=debug, batch_size=128, step=step)
    model.compile()
    # model.model.load_weights('title2')
    while (True):
        model.fit(2)
        model.check()
        # a = input('y or s?:')
        # if a == "s":
        #     model.model.save_weights('title2')
        # if a == "y":
        #     break
        
       

if __name__ == "__main__":
    main()
