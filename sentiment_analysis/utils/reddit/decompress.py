import bz2
import lzma
import zstandard as zstd
from driver import Driver
import os

class Decompress(object):

    def __init__(self):
        self.driver = Driver()
        self.driver.connect()

    def get_pending(self):
        result = self.driver.get_table_files()
        pending = [x for x in result if x[-2] == 1]
        if pending:
            idx = pending[0][0]
            compressed_file = pending[0][2]
            return idx, compressed_file
        else:
            return None, None
    
    def decompress(self):
        idx, compressed_file = self.get_pending()

        if idx:
            fname = compressed_file.split('/')[-1].split('.')[-2] + '.json'
            fd = '/'.join(compressed_file.split('/')[0:-1]) + '/' + fname
            if compressed_file.endswith('.bz2'):

                with open(fd, 'wb') as fo:
                    with open(compressed_file, 'rb') as fi:
                        z = bz2.BZ2Decompressor()
                        for block in iter(lambda: fi.read(blocksize), b''):
                            fo.write(z.decompress(block))

            elif compressed_file.endswith('.xz'):

                with open(fd, 'wb') as fo:
                    with open(compressed_file, 'rb') as fi:
                        z = lzma.LZMADecompressor()
                        for block in iter(lambda: fi.read(blocksize), b''):
                            fo.write(z.decompress(block))
                
            elif compressed_file.endswith('.zst'):

                with open(fd, 'wb') as fo:
                    with open(compressed_file, 'rb') as fi:
                        z = zstd.ZstdDecompressor()
                        reader = z.stream_reader(fi)
                        while True:
                            chunk = reader.read(16384)
                            if not chunk:
                                break
                            fo.write(chunk)
            else:
                raise NameError('{0} : format not allowed'.format(compressed_file))

            self.driver.set_table_files(idx, {'json': fd, 'state': 2})

            os.remove(compressed_file)

            self.driver.set_table_files(idx, {'zip': None})

            return True
        return False
       