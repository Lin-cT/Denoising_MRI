"""
This file is from https://github.com/msr-ai4science/distributed-training-talk/blob/main/utils/timer.py.

This timer helps to take measurements of the time taken by different parts of the code.
The timer can be used as a context manager, and it can be nested.

When used with CUDA available, it uses `torch.cuda.Event` to measure the time taken by the code.
This avoids synchronization between the GPU and CPU as much as possible.
See https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964 for more info.

Usage:

```python
timer = Timer()

with timer("outer"):
    for i in range(10):
        with timer("inner"):
            time.sleep(0.5)

print(timer.get_summary())
```

That should show something like

```
          name  mean_time_s  num_instances  fraction_of_runtime
0        outer     5.006001              1             0.999766
1  outer.inner     0.500558             10             0.999682
```

Metrics are *not* aggregated across workers.
"""

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd
import torch


@dataclass(slots=True)
class CudaTimer:
    start: torch.cuda.Event
    end: torch.cuda.Event
    name: str

    def elapsed_time_s(self):
        self.end.synchronize()
        miliseconds_per_second = 1000
        return self.start.elapsed_time(self.end) / miliseconds_per_second


@dataclass(slots=True)
class CpuTimer:
    start: float
    end: float | None
    name: str

    def elapsed_time_s(self) -> float:
        assert self.end is not None
        return self.end - self.start


class Timer:
    def __init__(self) -> None:
        self.timers: list[CudaTimer | CpuTimer] = []
        self.stack: deque[str] = deque()
        self.start_time = 0
        self.reset()

    def reset(self) -> None:
        assert len(self.stack) == 0, "There are still open timers."
        self.timers.clear()
        self.start_time = time.perf_counter()

    def get_summary(self) -> pd.DataFrame:
        assert len(self.stack) == 0, "There are still open timers."

        times: dict[str, list[float]] = defaultdict(list)
        for timer in self.timers:
            times[timer.name].append(timer.elapsed_time_s())

        # For each timer, aggregate the times and counts across workers.
        mean_times: dict[str, float] = {}
        counts: dict[str, int] = {}
        for timer_name, seconds in times.items():
            mean_times[timer_name] = torch.tensor(seconds, dtype=torch.float32).mean().item()
            counts[timer_name] = len(seconds)

        rec = pd.DataFrame(
            [
                {
                    "name": name,
                    "mean_time_s": mean,
                    "num_instances": counts[name],
                }
                for name, mean in mean_times.items()
            ]
        )

        if len(rec) == 0:
            return pd.DataFrame()

        rec["fraction_of_runtime"] = rec["mean_time_s"] * rec["num_instances"] / self.elapsed_time_since_reset_s()

        rec = rec.sort_values("fraction_of_runtime", ascending=False)

        return rec

    @contextmanager
    def __call__(self, name: str):
        assert self._is_valid_name(name)
        self.stack.append(name)
        try:
            name_path = ".".join(self.stack)
            if torch.cuda.is_available():
                with self._cuda_timer(name_path, name) as timer:
                    self.timers.append(timer)
                    yield
            else:
                with self._cpu_timer(name_path) as timer:
                    self.timers.append(timer)
                    yield
        finally:
            self.stack.pop()

    @contextmanager
    def _cuda_timer(self, full_name: str, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timer = CudaTimer(start, end, full_name)

        start.record()
        try:
            with torch.cuda.nvtx.range(name):
                yield timer
        finally:
            end.record()

    @contextmanager
    def _cpu_timer(self, name: str):
        start = time.perf_counter()
        timer = CpuTimer(start, None, name)
        try:
            yield timer
        finally:
            timer.end = time.perf_counter()

    def elapsed_time_since_reset_s(self) -> float:
        return time.perf_counter() - self.start_time

    def _is_valid_name(self, name: str) -> bool:
        return "." not in name  # The period is reserved as a symbol for nesting timers.
