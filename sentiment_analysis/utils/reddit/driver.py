from sqlalchemy.engine import create_engine
from sqlalchemy import Column, Table, MetaData
from sqlalchemy import Integer, Text
from sqlalchemy.dialects.postgresql import JSON, JSONB

import sqlalchemy

from utils import get_filenames

from sqlalchemy import and_

class Driver(object):

    def __init__(self):
        self.db = None
        self.engine = None
        self.meta = None

        self.address = 'postgresql+psycopg2://postgres:docker@localhost/postgres?port=5432'
        self.table_files_name = 'files'
        self.table_data_name = 'data'
        table_files_args = lambda: (
              Column('id', Integer, primary_key=True),
              Column('url', Text),
              Column('zip', Text),
              Column('json', Text),
              Column('table', Text),
              Column('state', Integer)
              )

        self.table_files = lambda x: Table(self.table_files_name, x, *table_files_args() )
        self.table_files_query = sqlalchemy.table(self.table_files_name, *table_files_args() )

    def connect(self):
        self.db = create_engine(self.address)
        self.engine = self.db.connect()
        self.meta = MetaData(self.engine)
        self.table_files(self.meta)
        self.meta.create_all()
        return self

    def init_files(self, url):
        file_list = get_filenames(url)[::-1]
        for idx, link in enumerate(file_list):
            params = {
                'url' : link, 
                'zip' : None,
                'json': None,
                'table': None,
                'state' : 0
            }
            statement = self.table_files_query.insert().values(**params)
            find = self.table_files_query.select().where(self.table_files_query.c.url == link)
            result = self.engine.execute(find).fetchall()
            if not result:
                self.engine.execute(statement)
        return self

    def get_table_files(self):
        find = self.table_files_query.select()
        return self.engine.execute(find).fetchall()

    def set_table_files(self, idx, params):
        query = sqlalchemy.update(self.table_files_query).where(self.table_files_query.c.id==idx).\
        values(**params)
        self.engine.execute(query)