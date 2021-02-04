# Context-aware Sequence Labeling for Condition Information Extraction from Historical Bridge Inspection Reports

This repository contains the trained model obtained from the paper Context-aware Sequence Labeling for Condition Information Extraction from Historical Bridge Inspection Reports. 

The model can be downloaded [here](https://virginia.box.com/s/ruvidulytbm6rh0ppm6p78y9nogb8r7f).

To use the model for tagging, `python run.py [modelPath] [inputPath]`

- Sentence to be tagged need be saved into an `input.txt` file. 
- The sentences will be tokenized by the nltk tokenizer and tagged by the trained model. Tokens with the same tag will be grouped into chunks for output.
