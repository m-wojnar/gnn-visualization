import time
from visutils import *

class Timer:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        print(self.name)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        print(f'Finished in {time.perf_counter() - self.start} s')
