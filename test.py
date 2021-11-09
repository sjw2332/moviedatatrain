import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="moviedatatrain/ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="moviedatatrain/ratings_test.txt")

train_data = pd.read_table('moviedatatrain/ratings_train.txt')
test_data = pd.read_table('moviedatatrain/ratings_test.txt')

print('훈련용 리뷰 개수 :',len(train_data))

print(train_data[:5])

print(train_data['document'].nunique())
print(train_data['label'].nunique())

train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
print('총 샘플의 수 :',len(train_data))

train_data['label'].value_counts().plot(kind = 'bar')
#plt.show()

print(train_data.groupby('label').size().reset_index(name = 'count'))

print(train_data.isnull().values.any())
print(train_data.isnull().sum())
print(train_data.loc[train_data.document.isnull()])

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인

print(len(train_data))

#okt = Okt()
print("="*100)
train_data['document'] = train_data['document'].str.replace(" ", "") # white space 데이터를 empty value로 변경
train_data['document'].replace("", np.nan, inplace=True)
print(train_data.isnull().sum())

print(train_data.loc[train_data.document.isnull()][:5])


test_data.drop_duplicates(subset = ['document'], inplace=True)
test_data['document'] = test_data['document'].str.replace(" ", "")
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))


#jpype 버전 맞춰도 에러,
okt = Okt()


#정수 인코딩
