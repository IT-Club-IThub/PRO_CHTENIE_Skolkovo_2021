from LSTM_NN import LSTM_NN
from BOW_NN import BOW_NN
from Embedding_NN import Embedding_NN
from RNN_NN import RNN_NN
import os

def main():
    for dataset_path in sorted(os.listdir('../datasets/'))[::-1]:
        print("--------------" + dataset_path.upper() + "--------------")
        dataset = '../datasets/' + dataset_path
        debug = False
        
        if "BOW" in dataset_path:
            print('BOW_NN')
            model = BOW_NN(dataset, debug=debug)
        else:
            print("LSTM_NN")
            model = LSTM_NN(dataset, debug=debug)

        try:
            model.compile()
            model.fit()
            model.check()
        except:
            print('Error')
       

if __name__ == "__main__":
    main()
  