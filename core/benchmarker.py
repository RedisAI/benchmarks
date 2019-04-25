from collections import defaultdict
from .config import config
from .reporter import Reporter
from .dockering import Dockering

# TODO: make sure imports are not affecting performance


class BenchmarkManager:
    _experiment_dict: dict = defaultdict(dict)

    def __init__(self):
        pass

    @classmethod
    def register(cls, name, fn):
        name_list = name.split('.')
        if len(name_list) != 3:
            raise Exception(
                'Naming should follow given format: <framework>.<modelname>.<entity>')
        framework, modelname, experiment = name_list
        if modelname != 'resnet':
            raise Exception('Only resnet is currently supported')
        if experiment in cls._experiment_dict.get(framework, {}).keys():
            raise Exception(f'Experiment with name {name} is already registered')
        cls._experiment_dict[framework][experiment] = fn

    @classmethod
    def run(cls):
        for backend, models in config['instances'].items():
            for m, experiments in models.items():
                for exp, devices in experiments.items():
                    for device, subconfigs in devices.items():
                        print(f'{backend}.{m}.{exp}.{device}')
                        subconfigs['exp_count'] = config['exp_count']
                        fn = cls._experiment_dict[backend][exp]
                        if exp == 'native':
                            fn(subconfigs, Reporter(backend, m, exp, device))
                        else:
                            if not subconfigs.get('docker'):
                                # TODO : remove this hack
                                continue
                            with Dockering(subconfigs['docker']) as server:
                                fn(subconfigs, Reporter(backend, m, exp, device))
        Reporter.summarize()
