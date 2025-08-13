# cf. https://medium.com/uncountable-engineering/pythons-line-profiler-32df2b07b290

from line_profiler import LineProfiler

profiler = LineProfiler()


def line_profile(func):
    def inner(*args, **kwargs):
        profiler.add_function(func)
        profiler.enable_by_count()
        return func(*args, **kwargs)

    return inner


def print_stats():
    profiler.print_stats()
