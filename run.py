import sys
import nltk

from BiLSTM import BiLSTM
from preprocessing import createMatrices


if len(sys.argv) < 2:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]

with open(inputPath, 'r') as f:
    text = f.read()

lstmModel = BiLSTM.loadModel(modelPath)

sentences = [{'tokens': nltk.word_tokenize(sent)} for sent in nltk.sent_tokenize(text)]
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)
tags = lstmModel.tagSentences(dataMatrix)

category_dict = {
    '0': 'Other', '3': 'Numerical Severity', '4': 'Qualitative Severity',
    '5': 'Location', '7': 'Condition Name'
}

for sentenceIdx in range(len(sentences)):
    print('='*10, 'sentence', sentenceIdx, '='*10)
    tokens = sentences[sentenceIdx]['tokens']
    labels = []
    for tokenIdx in range(len(tokens)):
        for modelName in sorted(tags.keys()):
            labels.append(tags[modelName][sentenceIdx][tokenIdx])

    currChunk, currLabel = tokens[0], labels[0]
    for t, l in zip(tokens[1:], labels[1:]):
        if currLabel != l:
            print(category_dict[currLabel], '--', currChunk)
            currChunk, currLabel = t, l
        else:
            currChunk += ' '
            currChunk += t
