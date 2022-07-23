import pandas as pd 
import fasttext 
import csv

w2v_model = fasttext.load_model("/workspace/datasets/fasttext/title_model.bin")

THRESHOLD = 0.75

top_words_pDF = pd.read_csv('/workspace/datasets/fasttext/top_words.txt', header=None, names=["tokens"])
synonyms = []

for index, row in top_words_pDF.iterrows():
    
    token = row['tokens']
    synonym_data = w2v_model.get_nearest_neighbors(token)
    nearest_neigh = [entry[1] for entry in synonym_data if entry[0] > THRESHOLD ]
    if len(nearest_neigh) > 1: 
        synonyms.append({"synonym": ','.join(nearest_neigh)})

synonym_pDF = pd.DataFrame(synonyms)
synonym_pDF.to_csv("/workspace/datasets/fasttext/synonyms.csv", header=False, index=False, sep = "\t", quoting = csv.QUOTE_NONE)
    




