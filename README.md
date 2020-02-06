Portuguese ELMo 
=================

This repository has the source code for the evaluation of Portuguese ELMo models published at 

Here's the source code for the model that has been submitted by the Deep Learning Brasil team to the 
[II Evaluation of Semantic Textual Similarity and Textual Inference in Portuguese](https://sites.google.com/view/assin2/english) 
that happened in 2019 during the [Symposium in Information and Human Language Technology](http://comissoes.sbc.org.br/ce-pln/stil2019/).

It achieved the best results among all submissions for the entailment task.

You may also want to read the current version of [our presentation slides](https://github.com/ruanchaves/assin/blob/master/STIL2019_presentation.pdf). A paper and a brief blog post on our findings are currently in the works.

## Installation

In order to reproduce our results, simply execute the commands below:

```
python3.6 -m venv assin2_env
source assin2_env/bin/activate
pip install -r requirements.txt
python assin-roberta.py settings.json
python assin-eval.py  assin2-test.xml ./submission/assin2/submission.xml
```

Depending on your resources you may want to edit `settings.json` and increase the amount of workers. Generally speaking, each worker will consume around 8 gigabytes of GPU memory. 

Increasing both the `buckets` and `kfold_buckets` parameters on the `settings.json` file by the same amount is expected to increase the accuracy of the model, although it will also proportionately increase the training time.


## Citation

A paper about our findings is planned to be released next year. 
Until then, you may cite this very repository: 

```
@misc{Rodrigues2019,
  author = {Ruan Chaves Rodrigues and Jéssica Rodrigues da Silva and Pedro Vitor Quinta de Castro and Nádia Félix Felipe da Silva and Anderson da Silva Soares},
  title = {ASSIN},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ruanchaves/assin}},
  commit = {f8c3f185fb3bcd106c8a0e5a12d9ef2c6119ec74}
}
```
