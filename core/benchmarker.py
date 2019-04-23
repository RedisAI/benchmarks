from collections import defaultdict


class BenchmarkManager:
    _experiment_dict: dict = defaultdict(dict)

    def __init__(self):
        pass

    @classmethod
    def register(cls, name, experiment):
        name_list = name.split('.')
        if len(name_list) < 3:
            raise Exception(
                'Naming should follow given format: <framework>.<modelname>.<entity>')
        framework, modelname, entity = name_list
        if modelname != 'resnet':
            raise Exception('Only resnet is currently supported')
        if entity in cls._experiment_dict.get(framework, {}).keys():
            raise Exception(f'Experiment with name {name} is already registered')

        # ignoring modelname for now
        cls._experiment_dict[framework][entity] = experiment

    @classmethod
    def run(cls):
        for framework, entities in cls._experiment_dict.items():
            for name, fn in entities.items():
                fn()
