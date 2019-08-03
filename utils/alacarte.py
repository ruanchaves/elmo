import yaml
import json

def load_yaml(fname):
    with open(fname, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    return f

if __name__ == '__main__':
    settings = load_yaml('../settings.yaml')
    fnames = [ settings['ASSIN']['files']['ptbr'],
    settings['ASSIN']['files']['pteu'] ]
    path = settings['ASSIN']['path']
    L = []
    for dct in fnames:
        for item in list(dct.values()):
            with open('.' + path + item, 'r') as f:
                tmp = json.load(f)
                L += tmp

out = []
for item in tmp:
    sentence = ' '.join(item['h'])
    out.append(sentence)
    sentence = ' '.join(item['t'])
    out.append(sentence)

out = '\n'.join(out)

with open('.' + path + 'alacarte.txt', 'w+') as f:
    print(out, file=f)