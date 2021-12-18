from LSTM_NN import LSTM_NN
from BOW_NN import BOW_NN
from Embedding_NN import Embedding_NN
from RNN_NN import RNN_NN
import os

def main():
    # for dataset_path in sorted(os.listdir('../datasets/'))[::-1]:
    for dataset_path in ['dataset_numWords_20000__xLen_500_step_100_BOW.npz', 'dataset_numWords_20000__xLen_500_step_100_.npz']:
        print("--------------" + dataset_path.upper() + "--------------")
        dataset = '../datasets/' + dataset_path
        debug = False
        try:
            model = BOW_NN(dataset, debug=debug, batch_size=128)
            model.compile()
            model.fit()
            model.check()
        except:
            print('Error')
       

if __name__ == "__main__":
    main()
  