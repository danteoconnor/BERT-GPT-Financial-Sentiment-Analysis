import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score

finance_news = pd.read_csv('all-data.csv', encoding='ISO-8859-1')
finance_news.head()

X = finance_news['Headline'].to_list()
y = finance_news['Sentiment'].to_list()

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

labels = {0:'neutral', 1:'positive',2:'negative'}

sent_val = list()
for x in X:
    inputs = tokenizer(x, return_tensors="pt", padding=True)
    outputs = finbert(**inputs)[0]

    val = labels[np.argmax(outputs.detach().numpy())]
    print(x, '----', val)
    print('#######################################################')
    sent_val.append(val)

print(accuracy_score(y, sent_val))
