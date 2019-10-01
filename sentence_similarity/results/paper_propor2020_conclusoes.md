# **Paper para o PROPOR2020**

( Vocês podem ignorar os arquivos .json nesta pasta. Somente os .csv's contém informação organizada ).

### ELMo

* ELMo_default : É o modelo ELMo que está disponível no AllenNLP.
* ELMo_brwac : ELMo treinado em um dataset maior ( BRWAC + wiki )
* ELMo_wiki: ELMo treinado somente na wiki, mas com perplexidade reduzida ( palavras que ocorrem 3 vezes ou menos foram eliminadas antes da etapa de treinamento ).

Para todos os modelos ELMo, foi considerado um embedding que consiste na concatenação dos três layers, sem realizar qualquer otimização dos embeddings para a tarefa em questão.

### Arquivos nesta pasta

* [Untitled.ipynb](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/Untitled.ipynb) : o notebook que gerou todos os csvs desta pasta a partir dos arquivos json. 

* [words_most_similar_to_unk](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/words_most_similar_to_unk.csv): resultado de uma chamada à função *most_similar()* no gensim, em *models.keyedvectors* . Mostra, para cada modelo, as palavras mais próximas ao embeddings para *unk*.

* [propor2020_test_results_ptbr](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/propor2020_test_results_ptbr.csv) e [propor2020_test_results_pteu](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/propor2020_test_results_pteu.csv): Mostra os resultados de todos os testes, ordenados por acurácia. Todos os arquivos que seguem abaixo foram construídos a partir deste. 

* [diff_preprocessing_ptbr](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/diff_preprocessing_ptbr.csv) e [diff_preprocessing_pteu](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/diff_preprocessing_pteu.csv) : A coluna **pearson_diff** registra, para cada modelo do NILC, para cada opção de usar ou não *unk* para tokens desconhecidos, a diferença de acurácia que ocorre entre fazer e não fazer o pré-processamento corretamente ( isto é, aplicar ao dataset o mesmo pré-processamento que foi utilizado no treinamento de cada modelo ).

* [unk_diff_not_preprocessed_ptbr](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/unk_diff_not_preprocessed_ptbr.csv) e [unk_diff_not_preprocessed_pteu](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/unk_diff_not_preprocessed_pteu.csv): **pearson_diff** registra, para os modelos do NILC, **sem** o pré-processamento correto, qual é a diferença de acurácia entre usar *unk* para palavras desconhecidas ou simplesmente ignorá-las. 

* [unk_diff_preprocessed_ptbr](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/unk_diff_preprocessed_ptbr.csv) e [unk_diff_preprocessed_pteu](https://github.com/ruanchaves/elmo/blob/master/sentence_similarity/results/unk_diff_preprocessed_pteu.csv): **pearson_diff** registra, para os modelos do NILC, **com** o pré-processamento correto, qual é a diferença de acurácia entre usar *unk* para palavras desconhecidas ou simplesmente ignorá-las. 

## Conclusões do paper


### Conclusões sobre embeddings clássicos 

* Para cada modelo, deve ser feito o mesmo pré-processamento do dataset que foi utilizado para treiná-lo. Caso contrário, a acurácia será afetada.

* Aumentar o tamanho do embedding significa aumento na acurácia, **exceto no caso do Wang2Vec** ( por quê? ).

* Utilizar *unk* como token para OOVs não é uma estratégia segura, principalmente quando *unk* está próximo de um cluster de embeddings muito significativo. No caso de embeddings que usam n-grams ( como o FastText ), a palavra *unk* está perto de palavras como *funk*, *punk* e palavras derivadas relacionadas à gêneros musicais. Isso pode prejudicar substancialmente a acurácia dos embeddings.

* Todos estes detalhes não foram levados em consideração no último paper publicado sobre os embeddings em questão ( https://arxiv.org/abs/1708.06025 ). Isso explica porque eles ficaram "surpresos": *" To our surprise, the word embedding models which achieved the best results on semantic analogies (see Table 2) were not the best in this semantic task. "* ( pg. 5 ). Isso leva o paper a chegar a uma conclusão equivocada: *"GloVe produced the best results for syntacticand semantic analogies, and the worst, **together with FastText,** for both POS tagging and sentence similarity. These results are aligned with those from [Faruqui et al. 2016], which suggest that word analogies are not appropriate for evaluating word embeddings."* Na realidade, após realizadas as devidas correções, o modelo que teve os melhores resultados nas word analogies ( FastText skip-gram ) é o mesmo modelo que teve os melhores resultados na similaridade semântica. Portanto, word analogies são medidas mais apropriadas em prever resultados de similaridade semântica do que pareceu para os autores deste paper.

* Se para a tarefa em questão for realmente necessário criar embeddings para OOVs, uma boa alternativa é utilizar a la carte embeddings (https://arxiv.org/abs/1805.05388 / https://github.com/NLPrinceton/ALaCarte / http://www.offconvex.org/2018/09/18/alacarte/ ), que possuem um custo computacional reduzido e resultados competitivos com abordagens mais pesadas ( ex. treinar uma RNN só para gerar embeddings para OOVs ).

### Conclusões sobre BERT e ELMo

* Entre os modelos ELMo, aquele que foi treinado com eliminação de palavras pouco comuns apresentou o melhor resultado.

* Embora BERT-multilingual sem fine-tuning e ELMo possuam resultados equivalentes para o dataset em português brasileiro ( BERT consegue 60% e ELMo consegue 61,8% ), o BERT-multilingual sofre perda de acurácia quando recebe o dataset em português europeu ( cai de 60% para 56% ), mas o ELMo consegue sustentar o mesmo nível de acurácia nos dois datasets. 

* Os melhores resultados possíveis são atingidos concatenando o melhor modelo ELMo com o maior modelo Word2Vec ( de tamanho 1000 ), conquistando uma acurácia de 65%. Os embeddings Word2Vec também se destacaram por terem sido capazes de sustentar a mesma acurácia tanto em português brasileiro quanto em portuguẽs europeu. 

* No geral, podemos concluir que o ELMo apresenta resultados consistentemente melhores do que qualquer modelo disponível, **inclusive o BERT-multilingual ( sem fine-tuning ).**