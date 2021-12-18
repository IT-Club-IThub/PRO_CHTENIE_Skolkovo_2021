from LSTM_NN import LSTM_NN
from BOW_NN import BOW_NN
from Embedding_NN import Embedding_NN
from RNN_NN import RNN_NN
import os
from datetime import datetime
import json
from get_xTest import get_xTest

def main():
    print(os.listdir('..'))
    dataset_path = "NEWNEWdataset_numWords_49208_xLen_300_step_300_.npz"
    print("--------------" + dataset_path.upper() + "--------------")
    dataset = '../datasets/' + dataset_path
    debug = True
    step = 300
    model = LSTM_NN(dataset, debug=debug, batch_size=128, step=step)
    model.compile()
    # model.model.load_weights('title2')
    model.fit(1)

    # открываем файл с id эссе и ответами
    with open('../../Satellites/test/test_task.json') as f_list:
        data = json.load(f_list)

        for i in range(len(data)):
            elem = data[i]
            xTestWrode = get_xTest(elem)
            print(xTestWrode)
            pred = model.model.predict(xTestWrode)
            print(pred)

            # elem["answer"] = model.check(get_xTest)
            
        # a = input('y or s?:')
        # if a == "s":
        #     model.model.save_weights('title2')
        # if a == "y":
        #     break
        
       

if __name__ == "__main__":
    main()
