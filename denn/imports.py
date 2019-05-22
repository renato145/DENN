import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from typing import Any, Callable, Collection, Optional, Union
from fastprogress import progress_bar, master_bar
from fastprogress.fastprogress import MasterBar, ProgressBar, format_time
from collections import Counter, defaultdict, Iterable
from numba import jit
from time import time as get_time
from IPython.display import clear_output

Path.ls = lambda x: list(x.iterdir())
PBar = Union[MasterBar, ProgressBar]
