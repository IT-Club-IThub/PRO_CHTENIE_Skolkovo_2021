from json.encoder import JSONEncoder
from LSTM_NN import LSTM_NN
from BOW_NN import BOW_NN
from Embedding_NN import Embedding_NN
from RNN_NN import RNN_NN
import os
from datetime import datetime
import json
from get_xTest import get_xTest
import random

def main():
    print(os.listdir('..'))
    #48937
    dataset_path = "NEWdataset_numWords_52628_xLen_300_step_300_.npz"
    # dataset_path = "NEWNEWdataset_numWords_52628_xLen_300_step_300_.npz"
    print("--------------" + dataset_path.upper() + "--------------")
    dataset = '../datasets/' + dataset_path
    debug = True
    step = 300
    model = RNN_NN(dataset, debug=debug, batch_size=128, step=step)
    model.compile()
    # model.model.load_weights('title2')
    model.fit(20)

    file = open('../../Satellites/test/test_task.json')
    data = json.load(file)
    file.close()
    length_data = len(data)
    for i in range(length_data):
        xTest = get_xTest(data[i])
        try:
            pred = model.model.predict(xTest)
        except:
            data[i]["answer"] = random.randint(0, 1)
            print(i, '/', length_data, "-", data[i]["answer"], '- error')
            continue
        data[i]["answer"] = int(pred[0][0] < pred[0][1])
        print(i, '/', length_data, "-", data[i]["answer"])

    file = open('../../Satellites/test/test_task_WE_ARE_THE_CEMPION.json', 'w')
    json.dump(data, file)
    file.close()

       

if __name__ == "__main__":
    main()
