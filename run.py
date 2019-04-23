from experiments import register_experiments
from core import benchmarker

register_experiments()

benchmarker.BenchmarkManager.run()
