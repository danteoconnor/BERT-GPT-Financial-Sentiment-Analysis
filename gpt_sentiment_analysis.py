import os
import numpy as np
import pandas as pd
import openai
from sklearn.metrics import accuracy_score

openai.api_key = "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"

finance_news = pd.read_csv('all-data.csv', encoding='ISO-8859-1')
finance_news.head()

X = finance_news['Headline'].to_list()
y = finance_news['Sentiment'].to_list()

labels = {0: 'neutral', 1: 'positive', 2: 'negative'}


def get_sentiment_gpt3(text):
    prompt = f"Sentiment of the following financial news headline: '{text}' is:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=5,
        n=1,
        stop=None,
        temperature=0.5,
    )

    result = response.choices[0].text.strip()
    if 'neutral' in result.lower():
        return 'neutral'
    elif 'positive' in result.lower():
        return 'positive'
    else:
        return 'negative'


sent_val = []
for x in X:
    val = get_sentiment_gpt3(x)
    print(x, '----', val)
    print('#######################################################')
    sent_val.append(val)

print(accuracy_score(y, sent_val))
