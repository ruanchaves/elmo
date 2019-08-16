from driver import Driver
from download import Download
from decompress import Decompress
from utils import load_yaml
import os
from ingest import Ingest
import multiprocessing as mp

def step_1(*args):
    d = Download(*args)
    while True:
        d.download()

def step_2(*args):
    d = Decompress()
    while True:
        d.decompress()


if __name__ == '__main__':
    os.chdir('..')
    os.chdir('..')
    settings = load_yaml('settings.yaml')
    url = settings['reddit']['pushshift']
    path = settings['reddit']['path']

    db = Driver()
    db.connect().init_files(url)
    db.engine.close()

    p1 = mp.Process(target=step_1, args=(0.75, path))
    p2 = mp.Process(target=step_2)
    procs = [p1,p2,p3]
    for p in procs:
        p.start()
    for p in procs:
        p.join()