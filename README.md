# Generating restaurant descriptions from structured input (work in progress)

A Natural Language Generation system for the [e2e-task](http://www.macs.hw.ac.uk/InteractionLab/E2E/). Currently this project is reimplementing a slightly modified version of the winning system by [Juraska et al.](https://arxiv.org/abs/1805.06553), which is a seq2seq model with a bidirectional LSTM encoder and a deep LSTM decoder using attention.

The final goal of the project is to explore the task of content selection, that is choosing which slots to include in an utterance, by learning the joint distribution of slots and sampling from them

## TODO
- [x] Implement beam search
- [x] Use Juraska reranking
- [ ] Implement semantically conditioned LSTM
- [ ] Run sampling experiments

## Codebase

The code is based on this [tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention), for which the code is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). However, since the architecture and the use case are different, it has significanlty evolved. The implementation for attention and the training loop have remained the same.

## Training

The training of the model is done using the script `generator/train.py`. The trained model is saved in checkpoints in the folder `generator/training_checkpoints`. 

## Generating utterances

Utterance generation is done using the script `generator/generate.py`. You need the file `training_info.pkl` created in training, which involves information such as model hyperparameters and dataset vocabulary.