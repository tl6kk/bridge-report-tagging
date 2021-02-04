"""
Modified from Nils Reimers
License: Apache-2.0
"""

import gc
import os
import sys
import time
import math
import keras
import random
import logging
import numpy as np
from keras.layers import *
from keras.optimizers import *
from keras.models import Model

from ChainCRF import ChainCRF
from sklearn.metrics import precision_score, recall_score, accuracy_score


class BiLSTM:
    def __init__(self, params=None):
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training
        self.models = None
        self.modelSavePath = None
        self.resultsSavePath = None
        # Hyperparameters for the network
        defaultParams = {'dropout': (0.5,0.5), 'classifier': ['CRF'], 'LSTM-Size': (100,), 'customClassifier': {},
                         'optimizer': 'adam',
                         'clipvalue': 0, 'clipnorm': 1,
                         'earlyStopping': 5, 'miniBatchSize': 32,
                         'featureNames': ['tokens', 'casing'], 'addFeatureDimensions': 10}
        if params != None:
            defaultParams.update(params)
        self.params = defaultParams

    def setMappings(self, mappings, embeddings):
        self.embeddings = embeddings
        self.mappings = mappings

    def setDataset(self, datasets, data):
        self.datasets = datasets
        self.data = data
        self.mainModelName = None
        self.epoch = 0
        self.learning_rate_updates = {}
        #'sgd': {1: 0.1, 3: 0.05, 5: 0.01}
        self.modelNames = list(self.datasets.keys())
        self.evaluateModelNames = []
        self.labelKeys = {}
        self.idx2Labels = {}
        self.trainMiniBatchRanges = None
        self.trainSentenceLengthRanges = None
        for modelName in self.modelNames:
            labelKey = self.datasets[modelName]['label']
            self.labelKeys[modelName] = labelKey
            self.idx2Labels[modelName] = {v: k for k, v in self.mappings[labelKey].items()}
            if self.datasets[modelName]['evaluate']:
                self.evaluateModelNames.append(modelName)
            logging.info("--- %s ---" % modelName)
            logging.info("%d train sentences" % len(self.data[modelName]['trainMatrix']))
            logging.info("%d dev sentences" % len(self.data[modelName]['devMatrix']))
            logging.info("%d test sentences" % len(self.data[modelName]['testMatrix']))

        if len(self.evaluateModelNames) == 1:
            self.mainModelName = self.evaluateModelNames[0]
        self.casing2Idx = self.mappings['casing']

    def buildModel(self):
        self.models = {}
        tokens_input = Input(shape=(None,), dtype='int32', name='words_input')
        tokens = Embedding(input_dim=self.embeddings.shape[0], output_dim=self.embeddings.shape[1], weights=[self.embeddings], trainable=True, name='word_embeddings')(tokens_input)

        inputNodes = [tokens_input]
        mergeInputLayers = [tokens]

        for featureName in self.params['featureNames']:
            if featureName == 'tokens':
                continue
            feature_input = Input(shape=(None,), dtype='int32', name=featureName+'_input')
            feature_embedding = Embedding(input_dim=len(self.mappings[featureName]), output_dim=self.params['addFeatureDimensions'], name=featureName+'_emebddings')(feature_input)
            inputNodes.append(feature_input)
            mergeInputLayers.append(feature_embedding)
        if len(mergeInputLayers) >= 2:
            merged_input = concatenate(mergeInputLayers)
        else:
            merged_input = mergeInputLayers[0]

        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:
            if isinstance(self.params['dropout'], (list, tuple)):
                shared_layer = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]))(shared_layer)
                # shared_layer = SimpleRNN(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1])(shared_layer)
                # shared_layer = LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1])(shared_layer)
            else:
                """ Naive dropout """
                shared_layer = Bidirectional(LSTM(size, return_sequences=True))(shared_layer)
                # shared_layer = LSTM(size, return_sequences=True)(shared_layer)
                # shared_layer = SimpleRNN(size, return_sequences=True)(shared_layer)
                if self.params['dropout'] > 0.0:
                    shared_layer = TimeDistributed(Dropout(self.params['dropout']), name='shared_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(shared_layer)
            cnt += 1

        for modelName in self.modelNames:
            output = shared_layer
            modelClassifier = self.params['customClassifier'][modelName] if modelName in self.params['customClassifier'] else self.params['classifier']
            if not isinstance(modelClassifier, (tuple, list)):
                modelClassifier = [modelClassifier]

            cnt = 1
            for classifier in modelClassifier:
                n_class_labels = len(self.mappings[self.labelKeys[modelName]])

                if classifier == 'Softmax':
                    output = TimeDistributed(Dense(n_class_labels, activation='softmax'), name=modelName+'_softmax')(output)
                    lossFct = 'sparse_categorical_crossentropy'
                elif classifier == 'CRF':
                    output = TimeDistributed(Dense(n_class_labels, activation=None),
                                             name=modelName + '_hidden_lin_layer')(output)
                    crf = ChainCRF(name=modelName+'_crf')
                    output = crf(output)
                    lossFct = crf.sparse_loss
                elif isinstance(classifier, (list, tuple)) and classifier[0] == 'LSTM':

                    size = classifier[1]
                    if isinstance(self.params['dropout'], (list, tuple)):
                        output = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name=modelName+'_varLSTM_'+str(cnt))(output)
                    else:
                        """ Naive dropout """
                        output = Bidirectional(LSTM(size, return_sequences=True), name=modelName+'_LSTM_'+str(cnt))(output)
                        if self.params['dropout'] > 0.0:
                            output = TimeDistributed(Dropout(self.params['dropout']), name=modelName+'_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(output)
                else:
                    assert(False) #Wrong classifier

                cnt += 1

            # :: Parameters for the optimizer ::
            optimizerParams = {}
            if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
                optimizerParams['clipnorm'] = self.params['clipnorm']

            if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
                optimizerParams['clipvalue'] = self.params['clipvalue']

            if self.params['optimizer'].lower() == 'adam':
                opt = Adam(**optimizerParams)
            elif self.params['optimizer'].lower() == 'nadam':
                opt = Nadam(lr=self.params['lr'], **optimizerParams)
            elif self.params['optimizer'].lower() == 'rmsprop':
                opt = RMSprop(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adadelta':
                opt = Adadelta(**optimizerParams)
            elif self.params['optimizer'].lower() == 'adagrad':
                opt = Adagrad(**optimizerParams)
            elif self.params['optimizer'].lower() == 'sgd':
                opt = SGD(lr=self.params['lr'], momentum=0.9, decay=0.0001, **optimizerParams)

            model = Model(inputs=inputNodes, outputs=[output])
            model.compile(loss=lossFct, optimizer=opt)

            model.summary(line_length=200)
            #logging.info(model.get_config())
            #logging.info("Optimizer: %s - %s" % (str(type(model.optimizer)), str(model.optimizer.get_config())))

            self.models[modelName] = model

    def trainModel(self):
        self.epoch += 1
        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:
            logging.info("Update Learning Rate to %f" % (self.learning_rate_updates[self.params['optimizer']][self.epoch]))
            for modelName in self.modelNames:
                K.set_value(self.models[modelName].optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch])
        for batch in self.minibatch_iterate_dataset():
            for modelName in self.modelNames:
                nnLabels = batch[modelName][0]
                nnInput = batch[modelName][1:]
                self.models[modelName].train_on_batch(nnInput, nnLabels)

    def minibatch_iterate_dataset(self, modelNames = None):
        """ Create based on sentence length mini-batches with approx. the same size. Sentences and
        mini-batch chunks are shuffled and used to the train the model """

        if self.trainSentenceLengthRanges == None:
            """ Create mini batch ranges """
            self.trainSentenceLengthRanges = {}
            self.trainMiniBatchRanges = {}
            for modelName in self.modelNames:
                trainData = self.data[modelName]['trainMatrix']
                trainData.sort(key=lambda x:len(x['tokens'])) #Sort train matrix by sentence length
                trainRanges = []
                oldSentLength = len(trainData[0]['tokens'])
                idxStart = 0
                #Find start and end of ranges with sentences with same length
                for idx in range(len(trainData)):
                    sentLength = len(trainData[idx]['tokens'])

                    if sentLength != oldSentLength:
                        trainRanges.append((idxStart, idx))
                        idxStart = idx

                    oldSentLength = sentLength
                #Add last sentence
                trainRanges.append((idxStart, len(trainData)))
                #Break up ranges into smaller mini batch sizes
                miniBatchRanges = []
                for batchRange in trainRanges:
                    rangeLen = batchRange[1]-batchRange[0]

                    bins = int(math.ceil(rangeLen/float(self.params['miniBatchSize'])))
                    binSize = int(math.ceil(rangeLen / float(bins)))

                    for binNr in range(bins):
                        startIdx = binNr*binSize+batchRange[0]
                        endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                        miniBatchRanges.append((startIdx, endIdx))
                self.trainSentenceLengthRanges[modelName] = trainRanges
                self.trainMiniBatchRanges[modelName] = miniBatchRanges

        if modelNames == None:
            modelNames = self.modelNames
        #Shuffle training data
        for modelName in modelNames:
            #1. Shuffle sentences that have the same length
            x = self.data[modelName]['trainMatrix']
            for dataRange in self.trainSentenceLengthRanges[modelName]:
                for i in reversed(range(dataRange[0]+1, dataRange[1])):
                    # pick an element in x[:i+1] with which to exchange x[i]
                    j = random.randint(dataRange[0], i)
                    x[i], x[j] = x[j], x[i]
            #2. Shuffle the order of the mini batch ranges
            random.shuffle(self.trainMiniBatchRanges[modelName])

        #Iterate over the mini batch ranges
        if self.mainModelName != None:
            rangeLength = len(self.trainMiniBatchRanges[self.mainModelName])
        else:
            rangeLength = min([len(self.trainMiniBatchRanges[modelName]) for modelName in modelNames])

        batches = {}
        for idx in range(rangeLength):
            batches.clear()
            for modelName in modelNames:
                trainMatrix = self.data[modelName]['trainMatrix']
                dataRange = self.trainMiniBatchRanges[modelName][idx % len(self.trainMiniBatchRanges[modelName])]
                labels = np.asarray([trainMatrix[idx][self.labelKeys[modelName]] for idx in range(dataRange[0], dataRange[1])])
                labels = np.expand_dims(labels, -1)
                batches[modelName] = [labels]
                for featureName in self.params['featureNames']:
                    inputData = np.asarray([trainMatrix[idx][featureName] for idx in range(dataRange[0], dataRange[1])])
                    batches[modelName].append(inputData)
            yield batches

    def storeResults(self, params, resultsFilepath):
        if resultsFilepath != None:
            directory = os.path.dirname(resultsFilepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.resultsSavePath = open(resultsFilepath, 'w')
            self.resultsSavePath.write('\t'.join([str(params[param]) for param in params]))
            self.resultsSavePath.write('\n')
            self.resultsSavePath.write('\t'.join(['train_score', 'dev_score', 'test_score',
                                                'max_dev_score', 'max_test_score',
                                                'dev_acc', 'dev_pre', 'dev_rec', 'dev_f1']))
            self.resultsSavePath.write('\n')
        else:
            self.resultsSavePath = None

    def fit(self, epochs):
        if self.models is None:
            self.buildModel()
        total_train_time = 0
        max_dev_score = {modelName:0 for modelName in self.models.keys()}
        max_test_score = {modelName:0 for modelName in self.models.keys()}
        no_improvement_since = 0
        result = dict()
        result['dev_f1'] = []
        result['test_f1'] = []
        result['train_f1'] = []
        result['test_pre'] = []
        result['test_rec'] = []
        max_epoch = 0

        for epoch in range(epochs):
            sys.stdout.flush()
            logging.info("\n--------- Epoch %d -----------" % (epoch+1))
            start_time = time.time()
            self.trainModel()
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            start_time = time.time()
            for modelName in self.evaluateModelNames:
                logging.info("-- %s --" % (modelName))
                dev_score, test_score, train_score = self.computeAccScores(modelName, self.data[modelName]['devMatrix'], self.data[modelName]['testMatrix'], self.data[modelName]['trainMatrix'])
                train_acc, train_pre, train_rec, train_f1, train_labels, train_predicts = self.computePre(modelName, self.data[modelName]['trainMatrix'])
                dev_acc, dev_pre, dev_rec, dev_f1, dev_labels, dev_predicts = self.computePre(modelName, self.data[modelName]['devMatrix'])
                test_acc, test_pre, test_rec, test_f1, test_labels, test_predicts = self.computePre(modelName, self.data[modelName]['testMatrix'])
                result['dev_f1'].append(dev_f1)
                result['test_f1'].append(test_f1)
                result['train_f1'].append(train_f1)
                result['test_pre'].append(test_pre)
                result['test_rec'].append(test_rec)
                if dev_score > max_dev_score[modelName]:
                    self.resultsSavePath.write(str(epoch)+'true\n')
                    self.resultsSavePath.write('\t'.join([str(l) for l in test_labels]))
                    self.resultsSavePath.write('\n'+str(epoch)+'pred\n')
                    self.resultsSavePath.write('\t'.join([str(p) for p in test_predicts]))
                    self.resultsSavePath.write('\n')
                    max_epoch = epoch
                    max_dev_score[modelName] = dev_score
                    max_test_score[modelName] = test_score
                    no_improvement_since = 0
                    #Save the model
                    if self.modelSavePath != None and dev_score > 0.95:
                        self.saveModel(modelName, epoch, dev_score, test_score)
                else:
                    no_improvement_since += 1

                if self.resultsSavePath != None:
                    self.resultsSavePath.write("\t".join(map(str, [epoch + 1, modelName, train_score, dev_score, test_score,
                                                                        max_dev_score[modelName], max_test_score[modelName],
                                                                        dev_acc, dev_pre, dev_rec, dev_f1,
                                                                        test_acc, test_pre, test_rec, test_f1])))
                    self.resultsSavePath.write("\n")
                    self.resultsSavePath.flush()

                logging.info("Max: %.4f dev; %.4f test" % (max_dev_score[modelName], max_test_score[modelName]))
                logging.info("")

            logging.info("%.2f sec for evaluation" % (time.time() - start_time))

            if self.params['earlyStopping']  > 0 and no_improvement_since >= self.params['earlyStopping']:
                logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
                break
        train, dev, test = result['train_f1'][max_epoch], result['dev_f1'][max_epoch], result['test_f1'][max_epoch]
        pre, rec = result['test_pre'][max_epoch], result['test_rec'][max_epoch]
        return train, dev, test, pre, rec

    def tagSentences(self, sentences):
        labels = {}
        for modelName, model in self.models.items():
            paddedPredLabels = self.predictLabels(model, sentences)
            predLabels = []
            for idx in range(len(sentences)):
                unpaddedPredLabels = []
                for tokenIdx in range(len(sentences[idx]['tokens'])):
                    if sentences[idx]['tokens'][tokenIdx] != 0:  # Skip padding tokens
                        unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])

                predLabels.append(unpaddedPredLabels)

            idx2Label = self.idx2Labels[modelName]

            labels[modelName] = [[idx2Label[tag] for tag in tagSentence] for tagSentence in predLabels]

        return labels

    def getSentenceLengths(self, sentences):
        sentenceLengths = {}
        for idx in range(len(sentences)):
            sentence = sentences[idx]['tokens']
            if len(sentence) not in sentenceLengths:
                sentenceLengths[len(sentence)] = []
            sentenceLengths[len(sentence)].append(idx)

        return sentenceLengths

    def predictLabels(self, model, sentences):
        predLabels = [None]*len(sentences)
        sentenceLengths = self.getSentenceLengths(sentences)

        for indices in sentenceLengths.values():
            nnInput = []
            for featureName in self.params['featureNames']:
                inputData = np.asarray([sentences[idx][featureName] for idx in indices])
                nnInput.append(inputData)

            predictions = model.predict(nnInput, verbose=False)
            predictions = predictions.argmax(axis=-1) #Predict classes
            predIdx = 0
            for idx in indices:
                predLabels[idx] = predictions[predIdx]
                predIdx += 1
        return predLabels

    def computeAccScores(self, modelName, devMatrix, testMatrix, trainMatrix):
        dev_acc = self.computeAcc(modelName, devMatrix)
        test_acc = self.computeAcc(modelName, testMatrix)
        train_acc = self.computeAcc(modelName, trainMatrix)

        logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
        logging.info("Test-Data: Accuracy: %.4f" % (test_acc))
        logging.info("Train-Data: Accuracy: %.4f" % (train_acc))
        return dev_acc, test_acc, train_acc

    def computeAcc(self, modelName, sentences):
        correctLabels = [sentences[idx][self.labelKeys[modelName]] for idx in range(len(sentences))]
        predLabels = self.predictLabels(self.models[modelName], sentences)

        numLabels = 0
        numCorrLabels = 0
        for sentenceId in range(len(correctLabels)):
            for tokenId in range(len(correctLabels[sentenceId])):
                numLabels += 1
                if correctLabels[sentenceId][tokenId] == predLabels[sentenceId][tokenId]:
                    numCorrLabels += 1


        return numCorrLabels/float(numLabels)

    def computePre(self, modelName, sentences):
        correctLabels = [sentences[idx][self.labelKeys[modelName]] for idx in range(len(sentences))]
        predLabels = self.predictLabels(self.models[modelName], sentences)
        labels = [y for x in correctLabels for y in x]
        predicts = [y for x in predLabels for y in x]
        accuracy = accuracy_score(labels, predicts)
        precision = precision_score(labels, predicts, average='macro')
        recall = recall_score(labels, predicts, average='macro')
        f1 = 2 * (precision * recall) / (precision + recall)
        return accuracy, precision, recall, f1, labels, predicts

    def saveModel(self, modelName, epoch, dev_score, test_score):
        import json
        import h5py

        if self.modelSavePath == None:
            raise ValueError('modelSavePath not specified.')
        savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[Epoch]", str(epoch+1)).replace("[ModelName]", modelName)
        directory = os.path.dirname(savePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.isfile(savePath):
            logging.info("Model "+savePath+" already exists. Model will be overwritten")
        self.models[modelName].save(savePath, True)
        with h5py.File(savePath, 'a') as h5file:
            h5file.attrs['mappings'] = json.dumps(self.mappings)
            h5file.attrs['params'] = json.dumps(self.params)
            h5file.attrs['modelName'] = modelName
            h5file.attrs['labelKey'] = self.datasets[modelName]['label']

    @staticmethod
    def loadModel(modelPath):
        import h5py
        import json
        from ChainCRF import create_custom_objects

        model = keras.models.load_model(modelPath, custom_objects=create_custom_objects())

        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
            params = json.loads(f.attrs['params'])
            modelName = f.attrs['modelName']
            labelKey = f.attrs['labelKey']

        bilstm = BiLSTM(params)
        bilstm.setMappings(mappings, None)
        bilstm.models = {modelName: model}
        bilstm.labelKeys = {modelName: labelKey}
        bilstm.idx2Labels = {}
        bilstm.idx2Labels[modelName] = {v: k for k, v in bilstm.mappings[labelKey].items()}
        return bilstm
