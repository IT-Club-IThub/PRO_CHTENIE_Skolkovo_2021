
import json
import numpy as np # Для работы с данными 
import sys

sys.path.append('..')

from tensorflow.keras.preprocessing.text import Tokenizer # Методы для работы с текстами

JSON_DATA_DIR = '../../Satellites'

def getSetFromIndexes(wordIndexes, xLen, step):
  xText = []
  wordsLen = len(wordIndexes) # Считаем количество слов
  index = 0 # Задаем начальный индекс 

  while (index + xLen <= wordsLen): # Идём по всей длине вектора индексов
    xText.append(wordIndexes[index:index+xLen]) # "Откусываем" векторы длины xLen
    index += step # Смещаемся вперёд на step
    
  return xText


def createSetsMultiClassesTest(wordIndexes, xLen, step): # Функция принимает последовательность индексов, размер окна, шаг окна
  classesXSamples = []        # Здесь будет список размером "кол-во классов*кол-во окон в тексте*длину окна"
  for wI in wordIndexes:      # Для каждого текста выборки из последовательности индексов
    classesXSamples.append(getSetFromIndexes(wordIndexes[0], xLen, step))


  xSamples = np.array(classesXSamples[0])

  return xSamples


def get_xTest(elem):
  textClasses = ['', '']

  # открываем файл с id эссе и ответами
  with open(JSON_DATA_DIR + '/train/train_standart.json', 'r') as f_list:
    data = json.load(f_list)

# проходимся по каждому "блоку" с эссе
    for i in range(len(data)):
      elem1 = data[i]

      with open(JSON_DATA_DIR + f'/train/essays/{elem1["id"]}.json', 'r') as essay:
        file = json.load(essay)
        text = file['text']
        if elem1['answer'] == False:
          textClasses[0] += text
          textClasses[0] += '#'
        else:
          textClasses[1] += text
          textClasses[1] += '#'

  texts_false = textClasses[0].split("#")
  texts_true = textClasses[1].split("#")

  trainText = []
  trainText.append(' '.join(texts_false))
  trainText.append(' '.join(texts_true))

  tokenizer = Tokenizer(
              filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
              lower=True,
              split=' ',
              oov_token='unknown',
              char_level=False)

  tokenizer.fit_on_texts(trainText)   


  testText = []

  with open(JSON_DATA_DIR + f'/test/essays/{elem["id"]}.json', 'r') as essay:
      file = json.load(essay)
      text = file['text']
      testText.append(text)

  xLen = 300

  testTextArray = tokenizer.texts_to_sequences(testText[0])
  xTest = createSetsMultiClassesTest(testTextArray, xLen, xLen)

  return xTest