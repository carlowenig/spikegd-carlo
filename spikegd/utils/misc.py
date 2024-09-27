import multiprocessing.pool
from typing import Any, Callable, Iterable, Mapping, Sequence, Set, Sized

import jax
import multiprocess.pool
import numpy as np


def standardize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, jax.Array):
        # this will also convert jax scalars to regular Python values
        return value.tolist()
    elif isinstance(value, Mapping):
        return {k: standardize_value(v) for k, v in value.items()}
    elif isinstance(value, str):
        # make sure strings are not treated as sequences
        return value
    elif isinstance(value, Sequence):
        return [standardize_value(v) for v in value]
    elif isinstance(value, Set):
        return {standardize_value(v) for v in value}
    else:
        return value


def advanced_map[Arg, Result](
    func: Callable[[Arg], Result],
    arg_iter: Iterable[Arg],
    parallel=True,
    chunksize: int = 1,
    processes: int | None = None,
    progress=False,
    total: int | None = None,
    ordered=False,
    pool_factory: Callable[[int | None], Any] = multiprocess.pool.Pool,
) -> list[Result]:
    if progress and total is None and isinstance(arg_iter, Sized):
        total = len(arg_iter)

    if parallel:
        pool: multiprocessing.pool.Pool

        with pool_factory(processes) as pool:
            if progress:
                from tqdm import tqdm

                if ordered:
                    iterator = pool.imap(func, arg_iter, chunksize)
                else:
                    iterator = pool.imap_unordered(func, arg_iter, chunksize)

                return list(tqdm(iterator, total=total))
            else:
                return pool.map(func, arg_iter, chunksize)
    else:
        if progress:
            from tqdm import tqdm

            return list(map(func, tqdm(arg_iter, total=total)))
        else:
            return list(map(func, arg_iter))
