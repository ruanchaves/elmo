Portuguese Language Models and Word Embeddings
=================

This is the source code for the architecture that generated the evaluation results mentioned in the paper *Portuguese Language Models and Word Embeddings: Evaluating on Semantic Similarity Tasks*. It's designed to evaluate all word embeddings from [nathanshartmann/portuguese_word_embeddings](https://github.com/nathanshartmann/portuguese_word_embeddings) on the semantic textual similarity tasks of the [ASSIN datasets](https://github.com/erickrf/assin) and also compare them with the results achieved by ELMo and BERT. Some of our tests will concatenate ELMo and word embeddings from the said repository.

## Benchmarks

Our full benchmarks are available under `reports/evaluation.csv` and also [on the slides]() of our presentation at the 14th edition of the International Conference on the Computational Processing of Portuguese (PROPOR 2020). The most relevant benchmarks for the semantic textual similarity task are reproduced below.

| Dataset           | Model                 | Embedding | Architecture | Dimensions |           PCC |           MSE |
|-------------------|-----------------------|-----------|--------------|------------|--------------:|--------------:|
| ASSIN 1 ( pt-BR ) | ELMo - wiki (reduced) |           |              |            |          0.62 |          0.47 |
|                   | ELMo - wiki (reduced) | word2vec  | CBOW         | 1000       |          0.62 |          0.47 |
|                   | portuguese-BERT       |           |              |            |          0.53 |          0.55 |
|                   | BERT-multilingual     |           |              |            |          0.51 |          1.94 |
| ASSIN 1 ( pt-PT ) | ELMo - wiki (reduced) |           |              |            |          0.63 |          0.73 |
|                   | ELMo - wiki (reduced) | word2vec  | CBOW         | 1000       |          0.64 |          0.73 |
|                   | portuguese-BERT       |           |              |            |          0.53 |          0.88 |
|                   | BERT-multilingual     |           |              |            |          0.52 |          0.90 |
| ASSIN 2           | ELMo - wiki (reduced) |           |              |            |          0.57 |          1.94 |
|                   | ELMo - wiki (reduced) | word2vec  | CBOW         | 1000       |          0.59 |          1.88 |
|                   | portuguese-BERT       |           |              |            |          0.64 |          1.69 |
|                   | BERT-multilingual     |           |              |            |          0.51 |          1.94 |

## Installation

Assuming you have installed Docker and nvidia-docker, the command below will reproduce all test results on this repository.

```
sudo bash scripts/quickstart.sh
```

Running this command will generate the `ruanchaves/elmo:2.0` docker image, if it doesn't exist yet, and also download all NILC embeddings, if they still haven't been downloaded to the `embeddings/NILC` folder.

Bear in mind that this script will also attempt run our private ELMo models: you can comment the relevant lines in case you don't have them, if you want to avoid error messages.

If you would also like to run BERT, extract your Tensorflow checkpoint files under the folder `embeddings/bert/portuguese`. It must be provided as a model checkpoint that can be understood by [bert-as-service](https://github.com/hanxiao/bert-as-service): you may have to rename some of the files in order to comply. Move `sentence_similarity/bert.yaml` to `settings/bert.yaml` and then recompile `scripts/quickstart.sh` by running `python generate_start.py`.

Your results will be stored in the folder `sentence_similarity/results` by default.

## Associated Repositories

You may want to take a look at the [ruanchaves/assin](https://github.com/ruanchaves/assin) repository. It contains tests which were performed with ensembles of fine-tuned Transformer models on the ASSIN datasets.