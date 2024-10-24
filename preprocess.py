import numpy as np
import csv

random_seed = 3
spambase = './dataset/spambase.csv'
isolet = './dataset/isolet.csv'

def readIsolet():
    dataset = []
    return_dataset = []
    with open(isolet) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float64)
    # numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict = []
        mydict.append(row[:-1])
        mydict.append(1 if row[-1] == 1 else 0)
        return_dataset.append(mydict)
    return return_dataset

def readSpambaseNormalized():
    dataset = []
    return_dataset = []
    with open(spambase) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float64)
    # numpy_dataset[:, :-1] = preprocessing.scale(numpy_dataset[:, :-1])
    for row in numpy_dataset:
        mydict =[]
        mydict.append(row[:-1])
        mydict.append(row[-1])
        return_dataset.append(mydict)
    return return_dataset

def readcsv(file):
    dataset = []
    return_dataset = []
    with open(file) as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row)
    dataset = np.array(dataset)
    numpy_dataset = dataset.astype(np.float64)
    for row in numpy_dataset:
        mydict = []
        mydict.append(row[:-1])
        mydict.append(row[-1])
        return_dataset.append(mydict)
    return return_dataset

def removefeature(dataset, remove_rate):
    data_0 = []
    data_none = []
    for i in dataset:
        a = []
        b = []
        c = []
        d = []
        x=len(i)
        for j in i[0]:
            if np.random.random() < remove_rate and x>=1:
                a.append(None)
                c.append(0)
                x-=1

            else:
                a.append(j)
                c.append(j)
        b.append(a)
        b.append(i[1])
        d.append(c)
        d.append(i[1])
        data_none.append(b)
        data_0.append(d)
    return data_none,data_0


def removelable(dataset, remove_rate):
    data = []
    for i in dataset:
        b = []
        if np.random.random() < remove_rate:
            i[1] = None
        b.append(i[0])
        b.append(i[1])
        data.append(b)
    return data