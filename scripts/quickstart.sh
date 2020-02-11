cd "$(dirname "${BASH_SOURCE[0]}")
sudo TESTS=nilc_with_unk_part1.yaml RESULTS=result_nilc_with_unk_part1.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
sudo TESTS=elmo_nilc_custom1_part1.yaml RESULTS=result_elmo_nilc_custom1_part1.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
sudo TESTS=elmo.yaml RESULTS=result_elmo.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
sudo TESTS=nilc_part2.yaml RESULTS=result_nilc_part2.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
sudo TESTS=nilc_with_unk_part2.yaml RESULTS=result_nilc_with_unk_part2.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
sudo TESTS=elmo_nilc_custom1_part2.yaml RESULTS=result_elmo_nilc_custom1_part2.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
sudo TESTS=nilc_part1.yaml RESULTS=result_nilc_part1.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
sudo TESTS=elmo_nilc_custom2_part2.yaml RESULTS=result_elmo_nilc_custom2_part2.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
sudo TESTS=elmo_nilc_custom2_part1.yaml RESULTS=result_elmo_nilc_custom2_part1.json BERT_DIR=portuguese CUDA_VISIBLE_DEVICES=1 bash start.sh &
