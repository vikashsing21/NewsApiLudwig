from ludwig.api import LudwigModel
import pandas as pd

df = pd.read_csv('Tweets.csv')
print(df.head())

model_definition = {
    'input_features': [
        {'name': 'title',
        'type': 'text',
        'encoder': 'rnn',
        'cell_type': 'lstm',
        'biderectional': 'true',
        'level': 'word'}
    ],
    'output_features': [
        {'name': 'news_sentiment', 'type': 'category'}
    ]
}

print('creating model')
model = LudwigModel(model_definition)
print('training model')
train_stats = model.train(data_df=df)
model.close()
