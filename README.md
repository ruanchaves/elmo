Portuguese Language Models and Word Embeddings
=================

This is the source code for the architecture that generated the evaluation results mentioned in the paper *Portuguese Language Models and Word Embeddings: Evaluating on Semantic Similarity Tasks*. It's designed to evaluate all word embeddings from [nathanshartmann/portuguese_word_embeddings](https://github.com/nathanshartmann/portuguese_word_embeddings) and also compare them with the results achieved by ELMo and BERT. Some of our tests will concatenate ELMo and word embeddings from the said repository.

## Benchmarks

Our benchmarks are available under `reports/evaluation.csv` and also [on the slides]() of our presentation at the 14th edition of the International Conference on the Computational Processing of Portuguese (PROPOR 2020).

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