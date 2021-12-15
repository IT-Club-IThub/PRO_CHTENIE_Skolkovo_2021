class Config:
    # Для использования этого конфига, сделайте
    # from config import Config
    # А так же замените название файлов на примерно эти конструкции
    #
    # Пример:
    # Было: 
    #   trainWordIndexes = np.load('/colab/MyDrive/Notebooks/train_word_indexes.npy', allow_pickle=True)
    # Стало:
    #   trainWordIndexes = np.load(Config.TRAIN_WORD_INDEXES_FILENAME, allow_pickle=True)

    TRAIN_WORD_INDEXES_FILENAME = '../creation/train_word_indexes.npy'
    TEST_WORD_INDEXES_FILENAME = '../creation/test_word_indexes.npy'
    DATASETS_DIR = "../datasets/"
    DATASET_X_TRAIN = "../datasets/xTrain.npy"
    DATASET_X_TRAIN_BOW = "../datasets/xTrainBOW.npy"
    DATASET_X_TEST = "../datasets/xTest.npy"
    DATASET_X_TEST_BOW = "../datasets/xTestBOW.npy"
    DATASET_Y_TRAIN = "../datasets/yTrain.npy"
    DATASET_Y_TEST = "../datasets/yTest.npy"
    