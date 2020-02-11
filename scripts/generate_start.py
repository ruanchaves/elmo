import os

cuda = 1
bert = 'portuguese'

cmd_strings = ["cd \"$(dirname \"${BASH_SOURCE[0]}\")\""]

for root, dirs, files in os.walk("../settings", topdown=False):
   for name in files:
       cmd = "screen -dmS {4} bash -c 'sudo TESTS={0} RESULTS={1} BERT_DIR={2} CUDA_VISIBLE_DEVICES={3} bash start.sh'"
       result = 'result_' + name.rstrip('yaml') + 'json'
       screen_name = name.rstrip('.yaml')
       cmd_strings.append(cmd.format(name, result, bert, cuda, screen_name))

with open('quickstart.sh','w+') as f:
    print('\n'.join(cmd_strings), file=f)