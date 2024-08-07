import numpy as np
import h5py

proteins = []

with h5py.File('reduced_embeddings_file.h5', 'r') as f:
    for new_identifier in f.keys():
        proteins.append((new_identifier, f[new_identifier].attrs["original_id"], np.array(f[new_identifier])))

print(f"The first protein in the set was assigned the identifier: {proteins[0][0]}.")
print(f"The ID extracted from the FASTA header is: {proteins[0][1]}.")
print(f"The shape of the embedding is: {proteins[0][2].shape}.")

proteins

import pandas as pd
df = pd.DataFrame(proteins)

f = df[2]

f.explode()

df1 = f.apply(pd.Series)

df1.to_csv("word2vec_TR_pos_3mer.csv")

df2 = pd.read_csv('Target_pos.csv')

type(df2)

df3 = pd.concat([df1, df2], axis=1, join='inner')
display(df3)

df3.to_csv("word2vec_pos_3mer.csv")
