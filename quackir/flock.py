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

import duckdb
from ._base import SecretProvider


class FlockManager:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection | None = None,
        db_path: str = "duck.db",
        secret_type: SecretProvider = SecretProvider.OLLAMA,
        api_url: str = "127.0.0.1:11434",
        api_key: str | None = None,
        resource_name: str | None = None,
        api_version: str | None = None,
    ):
        """Create a FlockManager.

        Parameters
        ----------
        conn : duckdb.DuckDBPyConnection | None
            Existing connection (optional). If not provided a new one is opened at db_path.
        db_path : str
            Path to DuckDB file when creating a new connection.
        secret_type : SecretProvider
            One of SecretProvider.OLLAMA, SecretProvider.OPENAI, SecretProvider.AZURE.
        api_url : str
            For OLLAMA only.
        api_key : str | None
            For OPENAI and AZURE.
        resource_name : str | None
            For AZURE only.
        api_version : str | None
            For AZURE only.
        """
        self.conn = conn or duckdb.connect(db_path)
        self.db_path = db_path
        self.secret_type = secret_type
        self.secret_name = f"__default_{secret_type.name.lower()}"
        self.api_url = api_url
        self.api_key = api_key
        self.resource_name = resource_name
        self.api_version = api_version
        self.model_alias = None

        self.ensure_loaded()
        self.ensure_secret(
            secret_type=self.secret_type,
            api_url=self.api_url,
            api_key=self.api_key,
            resource_name=self.resource_name,
            api_version=self.api_version,
        )

    def ensure_loaded(self) -> None:
        self.conn.execute("INSTALL flock FROM community;")
        self.conn.execute("LOAD flock;")

    def ensure_secret(
        self,
        *,
        secret_type: SecretProvider,
        api_url: str,
        api_key: str | None,
        resource_name: str | None,
        api_version: str | None,
    ) -> None:
        provider = secret_type.value
        params = {"TYPE": provider}

        if secret_type == SecretProvider.OLLAMA:
            if not api_url:
                raise ValueError("api_url required for OLLAMA secret")
            params["API_URL"] = api_url

        elif secret_type == SecretProvider.OPENAI:
            if not api_key:
                raise ValueError("api_key required for OPENAI secret")
            params["API_KEY"] = api_key

        elif secret_type == SecretProvider.AZURE:
            if not api_key or not resource_name or not api_version:
                raise ValueError("Missing required AZURE secret parameters")
            params.update(
                {
                    "API_KEY": api_key,
                    "RESOURCE_NAME": resource_name,
                    "API_VERSION": api_version,
                }
            )
        else:
            raise ValueError(f"Unsupported secret_type: {secret_type}")

        self.conn.execute(f'DROP SECRET IF EXISTS "{self.secret_name}";')
        fields = ", ".join(f"{k} ?" for k in params)
        values = list(params.values())
        self.conn.execute(f'CREATE SECRET "{self.secret_name}" ({fields});', values)

    def create_model(
        self,
        alias: str,
        provider_model: str,
        provider: str = "ollama",
        options_json: str = "{}",
        skip_if_exists: bool = True,
    ) -> None:
        self.model_alias = alias

        try:
            self.conn.execute(
                f"""
                CREATE MODEL(
                  '{alias}',
                  '{provider_model}',
                  '{provider}',
                  {options_json}
                )
                """
            )
        except Exception as e:
            msg = str(e)
            # Silently skip if the model alias already exists and skip_if_exists is True
            if ("Duplicate key" in msg or "already exists" in msg) and skip_if_exists:
                return
            raise

    def create_embedding(
        self,
        dest_table: str,
        file_path: str,
        id_column: str = "id",
        contents_column: str = "contents",
        embedding_dim: int = 768,
    ):
        model_cfg = (
            f"{{'model_name': '{self.model_alias}', 'secret': '{self.secret_name}'}}"
        )

        input_cfg = f"{{'context_columns': [ {{'data': {contents_column}}} ]}}"

        if file_path.endswith(".jsonl"):
            base_query = f"SELECT {id_column} AS id, {contents_column} FROM read_json_auto('{file_path}')"
        elif file_path.endswith(".parquet"):
            base_query = f"SELECT {id_column} AS id, {contents_column} FROM read_parquet('{file_path}')"
        else:
            raise ValueError("Unsupported file type (use .jsonl or .parquet)")

        sql_insert = f"""
            INSERT INTO {dest_table}
            SELECT
                id,
                CAST(
                    llm_embedding(
                        {model_cfg},
                        {input_cfg}
                    ) AS DOUBLE[{embedding_dim}]
                ) AS embedding
            FROM ({base_query})
            """
        self.conn.execute(sql_insert)

    def search_embedding(
        self,
        query_text: str,
        table_name,
        top_n=5,
        embedding_dim: int = 768,
    ):
        model_cfg = (
            f"{{'model_name': '{self.model_alias}', 'secret': '{self.secret_name}'}}"
        )

        sql_stmt = f"""
            WITH q AS (
                SELECT CAST(
                    llm_embedding(
                        {model_cfg},
                        {{'context_columns':[{{'data': ?}}]}}
                    ) AS DOUBLE[{embedding_dim}]
                ) AS embedding
            )
            SELECT
                t.id,
                array_cosine_similarity(t.embedding, q.embedding) AS score
            FROM {table_name} AS t, q
            ORDER BY score DESC
            LIMIT {top_n}
            """

        return self.conn.execute(sql_stmt, [query_text]).fetchall()
