#
# QuackIR: Reproducible IR research in RDBMS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sqlite3
from ._base import Searcher
from quackir._base import SearchType

class SQLiteSearcher(Searcher):
    def __init__(self, db_path="sqlite.db"):
        self.conn = sqlite3.connect(db_path)

    def get_search_type(self, table_name: str) -> SearchType:
        cur = self.conn.execute(f'select * from {table_name}')
        names = list(map(lambda x: x[0], cur.description))
        if "contents" in names:
            return SearchType.SPARSE
        else:
            raise ValueError(f"Unknown search type for table {table_name}. Ensure it has a 'contents' column. SQLite only supports sparse search currently.")

    def fts_search(self, query_string, top_n=5, table_name="corpus"):
        query = f"""
        SELECT id, bm25(fts_{table_name})*-1 AS score
        FROM fts_{table_name}
        WHERE fts_{table_name} MATCH '{{}}'
        ORDER BY score DESC
        LIMIT {{}}
        """
        query_string = query_string.replace("'", "''").replace('"', '""')
        terms = query_string.split()
        escaped_terms = terms
        # make each term a string so that any special chars are escaped
        escaped_terms = [f'"{term}"' for term in escaped_terms]
        # allow matching of any of the terms, using + or AND will turn it into boolean AND retrieval
        res = " OR ".join(escaped_terms)
        query = query.format(res, top_n)
        return self.conn.execute(query).fetchall()
    
    def embedding_search(self, query_embedding: str, top_n=5, table_name="corpus"):
        pass

    def rrf_search(self, query_string: str, query_embedding: str, top_n=5, k=60, table_name="corpus"):
        pass