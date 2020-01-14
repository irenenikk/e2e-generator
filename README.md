# Generating restaurant descriptions from structured input (E2E shared task)

A Natural Language Generation system for the [e2e-task](http://www.macs.hw.ac.uk/InteractionLab/E2E/). A modified version of the winning system by [Juraska et al.](https://arxiv.org/abs/1805.06553), which is a seq2seq model with a bidirectional LSTM encoder and a deep LSTM decoder using attention. In order to ease training, the system is approximated by using a one-layer GRU in both the encoder and decoder.

```
Input: name[Alimentum], area[city centre], familyFriendly[no], near[Burger King]
Generated: Alimentum is a non-child friendly place in the city centre area near Burger King .
Reference: Alimentum is not family-friendly. Alimentum is in the city center and it is near Burger King.
```

This project explores the task of content selection, that is choosing which slots to include in an utterance, by learning the joint distribution of slots and sampling from them. The joint distribution is modelled as a Bayesian network of binary variables.

## Requirements

Check the requirements file. Note that the project uses Tensorflow 2.0.

## Description of the system

The repository consists of three components:

- Seq2Seq neural network system used to generate utterances from MR slot information.
- A module that learns a Bayesian network for binary variables from data, and samples from the joint distribution.
- A module that creates a new training dataset by removing unrealised slots from the input slot description.

You can find the model code for the encoder, decoder and the attention module from `generator/models.py`.

## Codebase

The code is based on this [tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention), for which the code is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). However, since the architecture and the use case are different, it has significanlty evolved. The implementation for attention and the training loop for example have remained the same.

Some parts of the implementation of slug2slug are used for slot alignment. The [slug2slug project](https://github.com/jjuraska/slug2slug) is licensed under MIT, and a version can be found from this repo.

## Training

The training of the model is done using the script `generator/train.py`. The trained model is saved in checkpoints in the folder `generator/training_checkpoints`, and some necessary information is stored in training_info.pkl.

```
$ python generator/train.py rest_e2e/trainset.csv rest_e2e/development.csv -id <model identifier>
```

The model identifier must be unique as it is used to store the training checkpoints and training information

Parameters:

- Only use a subset of the training data with `-num`.

- Specify the amount of epochs with `-e` (defaults to 5).

- Disable teacher forcing with `-tf`.

- Recover training from a checkpoint with the same id: `-r`.

## Generating utterances

Utterance generation is done using the script `generator/generate.py`. In addition to the checkpoints, you need the file `training_info.pkl` created in training, which involves information such as model hyperparameters and dataset vocabulary. If your filenames match the ones given in training you don't have to specify the path to the checkpoints or the training info file.

```
$ python generator/generate.py rest_e2e/devset.csv -id <identifier of model> 
```

The trained model and training info will be loaded based on the id.

Parameters:

- In order to just print out the utterances, use the flag `-p` in generation. Otherwise BLEU score will be estimated for a sample of the dataset.

- Beam width is specified with the flag `-b`, and if none is given, the model uses token sampling to create utterance. 

- In order to use content sampling, use flag `-s`. If you wish, specify the path to the conditional probability tables with `-cpd`.


## Learning the Bayesian network

```
$ python generator/trainset_analysis.py rest_e2e/trainset.csv 
```

Model will be stored as a pickle to the desired path, defaulting to `cpd_model.pkl`.

## Creating cleaned training set

```
$ python generator/trainset_cleaner.py rest_e2e/ trainset.csv
```

The cleaned dataset will be stored with the suffix `_cleaned`

Parameters:
- Force redoing analysis even if old one is found: `-i`.

