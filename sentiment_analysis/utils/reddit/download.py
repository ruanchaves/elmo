from driver import Driver
from utils import get_large_file
import os
import shutil

class Download(object):
    def __init__(self, limit, path):
        self.driver = Driver()
        self.driver.connect()
        self.limit = limit
        self.path = path

    def above_limit(self):
        total, used, free = shutil.disk_usage('.')
        percent_used = used / ( used + free )
        if percent_used > limit:
            return True
        else:
            return False

    def get_pending(self):
        result = self.driver.get_table_files()
        pending = [x for x in result if x[-1] == 0 and x[-2] == 0]
        if not self.above_limit():
            if pending:
                idx = pending[0][0]
                link = pending[0][1]
                return idx, link
            else:
                return None, None
        else:
            return None, None
    
    def download(self):
        idx, link = self.get_pending()
        if idx is not None:
            fname = link.split('/')[-1]
            get_large_file(link, self.path + fname)
            disk_size = int(os.path.getsize(self.path + fname) / 1e+9)
            self.driver.set_table_files(idx, {'zip': self.path + fname, 'state': 1, 'disk_size': disk_size })
            return True
        else:
            return False
