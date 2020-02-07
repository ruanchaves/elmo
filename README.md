Portuguese ELMo 
=================

## Installation

Assuming you have installed Docker and nvidia-docker, the command below will reproduce all test results on this repository.

```
BERT_DIR=multi_cased_L-12_H-768_A-12 bash scripts/start.sh
```

`BERT_DIR=multi_cased_L-12_H-768_A-12` means that the BERT model is under the folder `embeddings/bert/multi_cased_L-12_H-768_A-12`. It must be provided as a model checkpoint that can be understood by [bert-as-service](https://github.com/hanxiao/bert-as-service) .

Running this command will generate the `ruanchaves/elmo:2.0` docker image, if it doesn't exist yet, and also download all NILC embeddings, if they still haven't been downloaded to the `embeddings/NILC` folder.

After these startup steps have been performed, it will automatically start a [bert-as-service](https://github.com/hanxiao/bert-as-service) server and perform all tests specified on `sentence_similarity/tests.yaml`. The results will be saved to `sentence_similarity/results/stats.json`.