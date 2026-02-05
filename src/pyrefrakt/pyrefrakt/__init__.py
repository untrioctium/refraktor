from pathlib import Path

_module_dir = Path(__file__).parent
_repo_root = _module_dir.parent.parent.parent

import os
os.environ['PATH'] = str(_repo_root) + os.pathsep + os.environ.get('PATH', '')
os.add_dll_directory(str(_repo_root))

from ._pyrefrakt import *

ASSETS_DIR = str(_repo_root / "assets")
CONFIG_DIR = str(_repo_root / "config")