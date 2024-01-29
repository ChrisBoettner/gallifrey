import contextlib

import joblib
from beartype.typing import Any
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Any:
    """Context manager to patch joblib to report
    into tqdm progress bar given as argument, from
    https://stackoverflow.com/a/58936697.

    Parameters
    ----------
    tqdm_object : tqdm
        The tqdm-wrapped object to iterate over, e.g.
        tqdm_joblib(tqdm(desc="My calculation", total=10).
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
