from ludwig.api import LudwigModel
import pandas as pd

model = LudwigModel.load('./results/experiment_run/model')
df_data = pd.read_csv('test.csv')
df_pred = model.predict(data_df=df_data)
print('*******************************************************************************************************')
# res=df.append(prediction['news_sentiment_predictions'],ignore_index=True)
res = pd.concat([df_data['text'], df_pred['airline_sentiment_predictions']], axis=1)
res.to_csv('Result1.csv', index=False)
model.close()
