# BIOSCAN-5M

![Alt Text](barcode_bert.png)

###### <h3> Overview

We conduct experiments with the Barcode-BERT baseline [Paper](https://arxiv.org/abs/2311.02401) in two stages: 
- (1) Pretraining: In this initial phase, we undertake non-overlapping k-mer segmentation of the DNA sequences, 
with 50\% of tokens being masked. Each token undergoes indexing using a code book, projecting tokens to an index while masked 
tokens are projected to 0. These projections are subsequently encoded and fed into a 12-layer transformer. 
Finally, the representation vectors from the transformer are employed to classify masked tokens based on the token 
indices in a fully unsupervised manner. 
- (2) Fine-tuning: Transitioning to the second stage, fine-tuning occurs without masking. 
The high representation vectors of all tokens are passed to the transformer, followed by a global mean-pooling function. 
The resulting output is then classified by species ground-truth labels.