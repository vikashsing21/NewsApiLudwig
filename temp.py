import pandas as pd
data=pd.read_csv('Tweets.csv')
data.to_csv('TweetsData',encoding='utf-8-sig',index=False)