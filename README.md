# Generating restaurant descriptions from structured input (work in progress)

A Natural Language Generation system for the [e2e-task](http://www.macs.hw.ac.uk/InteractionLab/E2E/). Currently this project is reimplementing the winning system by [Juraska et al.](https://arxiv.org/abs/1805.06553), which is a seq2seq model with a bidirectional LSTM encoder and a deep LSTM decoder using attention.

The final goal of the project is to explore the task of content selection, that is choosing which slots to include in an utterance, by learning the joint distribution of slots and sampling from them

## TODO
- [ ] Incorporate slot alignment
- [ ] Implement beam search
- [ ] Implement utterance reranking
- [ ] Run sampling experiments

## Codebse

The code is based on this [tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention). However, since the architecture and the use case are different, it has significanlty evolved. The implementation for attention and the training loop have remained the same.
