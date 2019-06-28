import re, numpy as np, pandas as pd, matplotlib.pyplot as plt, concurrent
from dataclasses import dataclass
from pathlib import Path
from functools import partial
from typing import Any, Callable, Collection, Optional, Union, Dict, Tuple
from fastprogress import progress_bar, master_bar
from fastprogress.fastprogress import MasterBar, ProgressBar, format_time
from collections import Counter, defaultdict, Iterable
from numba import jit
from time import time as get_time
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from IPython.display import clear_output
from copy import deepcopy

Path.ls = lambda x: list(x.iterdir())
PBar = Union[MasterBar, ProgressBar]
