cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..

# server_timeout=30

# bert-serving-start -pooling_layer 11 -model_dir embeddings/bert/$BERT_DIR  &
# sleep $server_timeout

cd embeddings
python download.py

cd ..

cd sentence_similarity

# python preprocessing.py
python evaluate.py