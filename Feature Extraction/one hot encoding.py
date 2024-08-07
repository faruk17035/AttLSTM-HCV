import numpy as np
import pandas as pd
from collections import Counter

def get_data(file, k):
    fi = open(file, 'r')
    max_seq = 800
    seqs = []
    seq = ""
    labels = []

    for line in fi:
        if (line[0] == ">"):
            seqs.append(seq)
            seq = ""
        else:
            seq = seq + line.split('\n')[0]

    # Include the last sequence in seqs
    if seq:
        seqs.append(seq)

    fi.close()  # Close the file

    print(len(seqs))

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    seq_new = []
    for seq in seqs:
        if (len(seq) < max_seq):
            seq_new.append((seq + 'Z' * (max_seq - len(seq))))
        else:
            seq_new.append(seq[:max_seq])

    a = []
    b = []
    for seq in seq_new:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            kmer_encoding = [1 if amino in kmer else 0 for amino in amino_acids]
            a.extend(kmer_encoding)
        b.append(a)
        a = []

    data = np.array(b, dtype=int)

    print(data.shape)

    return data


def get_label(file):
    labels = []
    label = pd.read_csv(file, header=None).values
    for i in label:
        labels.append(int(i))
    labels = np.array(labels, dtype=int)
    return labels


if __name__ == '__main__':
    type = 'Train'
    #data_file = './Data/' + type + '/FULL_Train.fasta'
    #label_file = './Data/' + type + '/Train_True_Labels.csv'
    data_file = 'TR_pos.fasta'
    label_file = 'Target_pos.csv'
    
    # Adjust k to the desired k-mer length
    k = 3
    
    x_train = get_data(data_file, k)
    my_dataframe = pd.DataFrame(x_train)
    csv_file_path = f'TR_pos_kmer_{k}.csv'
    my_dataframe.to_csv(csv_file_path, index=False)
    
    y_train = get_label(label_file)
    print(len(x_train), len(y_train))
    print(Counter(y_train))