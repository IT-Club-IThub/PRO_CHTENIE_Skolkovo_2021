from LSTM_NN import LSTM_NN
from BOW_NN import BOW_NN
from Embedding_NN import Embedding_NN
from RNN_NN import RNN_NN
import os

def main():
    datasets = []
    for l in sorted(os.listdir('../datasets/')):
        if("BOW" not in l.upper()):
            datasets.append(l)

    for dataset_path in datasets[::-1]:
        print("--------------" + dataset_path.upper() + "--------------")
        dataset = '../datasets/' + dataset_path
        debug = False
        for step in dataset_path.split('_')[6].replace('(', '').replace(')', '').split(','):
            step = int(step)
            print(f'{step=}')
            model = LSTM_NN(dataset, debug=debug, batch_size=128, step=step)
            model.compile()
            model.fit()
            model.check()
        
       

if __name__ == "__main__":
    main()
  