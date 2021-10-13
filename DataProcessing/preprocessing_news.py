import os
import matplotlib.pyplot as plt
import numpy as np

from pyvi import ViTokenizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

INPUT = './data/news_vnexpress'
os.makedirs("images", exist_ok=True)
n = 0

# print(os.listdir(INPUT))
# print file name -> list

# print(os.path.join(INPUT, 'doi-song'))
# create path for each file

# for label in os.listdir(INPUT):
#     num = len(os.listdir(os.path.join(INPUT, label)))
#     print(f'{label} : {num}')
#     n += num

# print(f'Sum: {n}')

data_train = load_files(container_path=INPUT, encoding="utf-8")

# print(data_train.target_names)
# <=> os.listdir(INPUT)

# print(len(data_train.filenames))
# print(len(data_train.data))
# print(data_train.data[0:1])

with open("data/vietnamese-stopwords.txt", encoding="utf-8") as f:
    stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords] 
print(f"Số lượng stopwords: {len(stopwords)}")
# strip() bỏ dấu cách đầu và cuối câu

module_count_vector = CountVectorizer(stop_words= stopwords)
print(module_count_vector)



