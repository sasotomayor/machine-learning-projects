import pandas as pd

#df = pd.read_csv('reviews_train.tsv', sep='\t', encoding='ISO-8859-1')

df_2 = pd.read_csv("stopwords.txt", sep=" ", header=None)[0].tolist()
print(df_2)