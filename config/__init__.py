import os
from quackir._base import load_env

if not os.getenv("_ENV_LOADED"):
    load_env()
