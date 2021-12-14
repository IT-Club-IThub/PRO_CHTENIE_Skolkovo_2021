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
    