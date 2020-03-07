Portuguese Language Models and Word Embeddings
=================

This repository has primarily been designed to assess the quality of the [Portuguese ELMo representations made available through the AllenNLP library](https://allennlp.org/elmo) in comparison with the language models and word embeddings currently available for the Portuguese language.

This source code can reproduce the experiments mentioned in our paper [Portuguese Language Models and Word Embeddings: Evaluating on Semantic Similarity Tasks](https://www.springer.com/gp/book/9783030415044). It's designed to evaluate all word embeddings from [nathanshartmann/portuguese_word_embeddings](https://github.com/nathanshartmann/portuguese_word_embeddings) on the semantic textual similarity tasks of the [ASSIN datasets](https://github.com/erickrf/assin) and also compare them with the results achieved by ELMo and BERT. Some of our tests will concatenate ELMo and word embeddings from the said repository.

* [Paper](https://www.springer.com/gp/book/9783030415044)

* [Blog post](https://ruanchaves.github.io/portuguese-language-models/)

* [PROPOR 2020 Presentation](presentations/PROPOR_2020_presentation.pdf)

* [Benchmarks](reports/evaluation.csv)

## Benchmarks

Our full benchmarks are available under [`reports/evaluation.csv`](reports/evaluation.csv). The most relevant benchmarks for the semantic textual similarity task are reproduced below.

| Dataset           | Model                 | Embedding | Architecture | Dimensions |           PCC |           MSE |
|-------------------|-----------------------|-----------|--------------|------------|--------------:|--------------:|
| ASSIN 1 ( pt-BR ) | ELMo - wiki (reduced) |           |              |            |          0.62 |          0.47 |
|                   | ELMo - wiki (reduced) | word2vec  | CBOW         | 1000       |          0.62 |          0.47 |
|                   | [portuguese-BERT](https://github.com/neuralmind-ai/portuguese-bert)       |           |              |            |          0.53 |          0.55 |
|                   | [BERT-multilingual (cased)](https://github.com/google-research/bert/blob/master/multilingual.md)     |           |              |            |          0.51 |          1.94 |
| ASSIN 1 ( pt-PT ) | ELMo - wiki (reduced) |           |              |            |          0.63 |          0.73 |
|                   | ELMo - wiki (reduced) | word2vec  | CBOW         | 1000       |          0.64 |          0.73 |
|                   | [portuguese-BERT](https://github.com/neuralmind-ai/portuguese-bert)       |           |              |            |          0.53 |          0.88 |
|                   | [BERT-multilingual (cased)](https://github.com/google-research/bert/blob/master/multilingual.md)     |           |              |            |          0.52 |          0.90 |
| ASSIN 2           | ELMo - wiki (reduced) |           |              |            |          0.57 |          1.94 |
|                   | ELMo - wiki (reduced) | word2vec  | CBOW         | 1000       |          0.59 |          1.88 |
|                   | [portuguese-BERT](https://github.com/neuralmind-ai/portuguese-bert)       |           |              |            |          0.64 |          1.69 |
|                   | BERT-multilingual     |           |              |            |          0.51 |          1.94 |

In our benchmarks, the ELMo model labelled as `wiki` is the first public Portuguese ELMo model that was made available through the [AllenNLP library website](https://allennlp.org/elmo). Since then it has been replaced on the website by `wiki (reduced)`.

The `BRWAC` model was trained on [brWaC](https://www.researchgate.net/publication/326303825_The_brWaC_Corpus_A_New_Open_Resource_for_Brazilian_Portuguese), and the `wiki (reduced)` was trained on the same dataset as `wiki` after words with word frequency below four occurrences were eliminated from the dataset. 

## Installation

Assuming you have installed Docker and nvidia-docker, the command below will reproduce all test results on this repository.

```
sudo bash scripts/quickstart.sh
```

Running this command will generate the `ruanchaves/elmo:2.0` docker image, if it doesn't exist yet, and also download all NILC embeddings, if they still haven't been downloaded to the `embeddings/NILC` folder.

If you would also like to run BERT, extract your Tensorflow checkpoint files under the folder `embeddings/bert/portuguese`. It must be provided as a model checkpoint that can be understood by [bert-as-service](https://github.com/hanxiao/bert-as-service): you may have to rename some of the files in order to comply. Move `sentence_similarity/bert.yaml` to `settings/bert.yaml` and then recompile `scripts/quickstart.sh` by running `python generate_start.py`.

Your results will be stored in the folder `sentence_similarity/results` by default.

## Associated Repositories

* [Pull request to nathanshartmann/portuguese_word_embeddings: Improvements to the scores of evaluated embeddings #11](https://github.com/nathanshartmann/portuguese_word_embeddings/pull/11) 

* You may want to take a look at the [ruanchaves/assin](https://github.com/ruanchaves/assin) repository. It contains tests which were performed with ensembles of fine-tuned Transformer models on the ASSIN datasets.

## Citation

```
@inproceedings{rodrigues_propor2020,
  author = {Ruan Chaves Rodrigues and Jéssica Rodrigues da Silva and Pedro Vitor Quinta de Castro and Nádia Félix Felipe da Silva and Anderson da Silva Soares },
  title = {Portuguese Language Models and Word Embeddings: Evaluating on Semantic Similarity Tasks},
  editor = { Paulo Quaresma and Renata Vieira and Sandra Aluísio and Helena Moniz and Fernando Batista and Teresa Gonçalves },
  booktitle = { Computational Processing of the Portuguese Language },
  note = { 14th International Conference, PROPOR 2020, Evora, Portugal, March 2–4, 2020, Proceedings },
  publisher = { Springer International Publishing },
  address = { Springer Nature Switzerland AG },
  doi = {10.1007/978-3-030-41505-1},
  year = {2020}}
```
