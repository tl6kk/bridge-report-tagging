import os
import gc
import sys
import logging
import numpy as np
import random as rn
import tensorflow as tf
from keras import backend as K
from statistics import mean,stdev

from BiLSTM import BiLSTM
from preprocessing import perpareDataset, loadDatasetPickle


loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

trains, devs, tests, pres, recs = [], [], [], [], []
for i in range(1):
    name = 'fold_'+str(i)
    datasets = {
        name:
            {'columns': {0:'tokens', 1:'POS'},
             'label': 'POS',
             'evaluate': True,
             'commentSymbol': None}
    }

    embeddingsPath = 'embedding/komninos_english_embeddings.gz'
    pickleFile = perpareDataset(embeddingsPath, datasets)
    embeddings, mappings, data = loadDatasetPickle(pickleFile)
    print(embeddings.shape[0], embeddings.shape[1])

    params = {
        'lr': 0.001,
        'dropout': 0,
        'classifier': ['CRF'],
        'LSTM-Size': [100, 100],
        'optimizer': 'nadam',
        'charEmbeddings': None,
        'clipvalue': 5,
        'clipnorm': 1,
        'earlyStopping': 5,
        'miniBatchSize': 8,
        'featureNames': ['tokens'],
    }

    model = BiLSTM(params)
    print(model.params)
    model.setMappings(mappings, embeddings)
    model.setDataset(datasets, data)

    result_path = 'results/'+name+'_'.join([str(params[param]) for param in params])+'.txt'
    model.storeResults(params, result_path) #Path to store performance scores for dev / test
    model.modelSavePath = "models/[ModelName]_[DevScore]_[Epoch].h5" #Path to store models
    train, dev, test, pre, rec = model.fit(epochs=60)
    del model
    gc.collect()
