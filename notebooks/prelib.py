import sys
from pathlib import Path

if '..' not in sys.path:
    sys.path.append('..')


data_dir = Path('../data/').resolve()