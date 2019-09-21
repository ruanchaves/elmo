
import yaml
import sys
import hashlib
def load_yaml(fname):
    with open(fname, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return f

def sentence_to_hash(sentence, len_=16):
        return int(hashlib.md5(sentence.encode('utf-8')).hexdigest(), len_)