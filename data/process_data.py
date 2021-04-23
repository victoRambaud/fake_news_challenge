import csv
import numpy as np

def process_data(p_val=0.2):
    train = []
    val = []

    with open('train/full_train_stances.csv','r') as f:
        reader = csv.reader(f)
        next(reader)
        for l in reader:
            if np.random.rand() > p_val:
                train.append(l)
            else:
                val.append(l)

    print(len(train), len(val))

    for dat, fn in zip([train, val],['train/train_stances.csv','val/val_stances.csv']):
        with open(fn,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Headline','Body ID','Stance'])
            for l in dat:
                writer.writerow(l)

if __name__ == '__main__':
    process_data()