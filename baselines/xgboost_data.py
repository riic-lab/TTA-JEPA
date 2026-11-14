import numpy as np
import xgboost as xgb


class TaskIterator(xgb.DataIter):
    """Bounded feature chunks; each valid admission contributes unit task weight."""

    def __init__(self, cache, task, chunk, prefix):
        super().__init__(cache_prefix=str(prefix), release_data=True)
        self.cache, self.task, self.chunk = cache, task, chunk
        self.rows = np.flatnonzero(cache["valid"][:, task])
        self.counts = np.bincount(cache["admission"][self.rows])
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def next(self, input_data):
        rows = self.rows[self.cursor:self.cursor+self.chunk]
        if not len(rows):
            return False
        input_data(data=np.asarray(self.cache["states"][rows]), label=np.asarray(self.cache["labels"][rows, self.task]),
                   weight=(1 / self.counts[self.cache["admission"][rows]]).astype(np.float32))
        self.cursor += len(rows)
        return True
