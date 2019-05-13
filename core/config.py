import os
import argparse


# TODO: Move flask server files to docker

class ConfigManager:
    devices = ['cpu', 'gpu']
    backends = ['tensorflow', 'pytorch']
    exp_count = 50
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets = os.path.join(root, 'assets')
    images_path = os.path.join(assets, 'images/guitar.jpg')
    image_class = 402  # guitar
    tf_serving_path = os.path.join(assets, 'tf_serving_builds')
    # TODO: perhaps make it configurable
    tf_flask_path = os.path.join(root, 'experiments/_tensorflow/_flask')
    pt_flask_path = os.path.join(root, 'experiments/_pytorch/_flask')
    pt_grpc_path = os.path.join(root, 'experiments/_pytorch/_grpc_server')
    models = ['resnet']
    experiments = ['native', 'redisai', 'flask', 'grpc']
    flask_url = 'http://127.0.0.1:8000/predict'
    tfserving_grpc_url = 'localhost:8500'
    pt_grpc_url = 'localhost:50051'
    redisai_url = 'localhost:6379'

    def __init__(self, args):
        if args.backend != 'all':
            self.backends = [args.backend]
        if args.device != 'all':
            self.devices = [args.device]
        if args.exp != 'all':
            assert len(set(args.exp) - set(self.experiments)) == 0,\
                f'Given experiment is not defined: use from {self.experiments}'
            self.experiments = args.exp
        self.exp_count = args.count
        self.no_docker = args.no_docker

    def get_config_dict(self):
        # TODO: Move it to a config file
        temp = {
            'tensorflow': {
                'native': {
                    'cpu': {'modelpath': os.path.join(self.assets, 'resnet50.pb')},
                    'gpu': {}},
                'redisai': {
                    'cpu': {
                        'server': self.redisai_url,
                        'modelpath': os.path.join(self.assets, 'resnet50.pb'),
                        'docker': {
                            'image': 'tensorwerk/raibenchmarks:redisai-optim-cpu',
                            'ports': {6379: 6379}}},
                    'gpu': {}},
                'flask': {
                    'cpu': {
                        'server': self.flask_url,
                        'docker': {
                            'image': 'tensorwerk/raibenchmarks:flask-optim-cpu',
                            'volumes': {
                                self.assets: '/root/data',
                                self.tf_flask_path: '/root'},
                            'ports': {8000: 8000}}},
                    'gpu': {}},
                'grpc': {
                    'cpu': {
                        'server': self.tfserving_grpc_url,
                        'docker': {
                            'image': 'tensorwerk/raibenchmarks:tfserving-optim-cpu',
                            'ports': {8500: 8500, 8501: 8501},
                            # TODO: change this hardcoded resnet
                            'volumes': {self.tf_serving_path: '/models/resnet'},
                            'envs': {
                                'MODEL_NAME': 'resnet'}}},
                    'gpu': {}}},
            'pytorch': {
                'native': {
                    'cpu': {'modelpath': os.path.join(self.assets, 'resnet50.pt')},
                    'gpu': {}},
                'redisai': {
                    'cpu': {
                        'server': self.redisai_url,
                        'modelpath': os.path.join(self.assets, 'resnet50.pt'),
                        'docker': {
                            'image': 'tensorwerk/raibenchmarks:redisai-optim-cpu',
                            'ports': {6379: 6379}}},
                    'gpu': {}},
                'flask': {
                    'cpu': {
                        'server': self.flask_url,
                        'docker': {
                            'image': 'tensorwerk/raibenchmarks:flask-optim-cpu',
                            'volumes': {
                                self.assets: '/root/data',
                                self.pt_flask_path: '/root'},
                            'ports': {8000: 8000}}},
                    'gpu': {}},
                'grpc': {
                    'cpu': {
                        'server': self.pt_grpc_url,
                        'docker': {
                            'image': 'tensorwerk/raibenchmarks:grpc-cpu',
                            'ports': {50051: 50051},
                            'volumes': {
                                self.assets: '/root/data',
                                self.pt_grpc_path: '/root'}
                        }
                    },
                    'gpu': {}}},
            'onnx': {
                'native': {
                    'cpu': {},
                    'gpu': {}},
                'redisai': {
                    'cpu': {},
                    'gpu': {}},
                'flask': {
                    'cpu': {},
                    'gpu': {}},
                'grpc': {
                    'cpu': {},
                    'gpu': {}}}
        }
        out = {
            'instances': {},
            'exp_count': self.exp_count,
            'root': self.root,
            'assets': self.assets,
            'img_path': self.images_path,
            'img_class': self.image_class}
        for b in self.backends:
            out['instances'][b] = {}
            for m in self.models:
                out['instances'][b][m] = {}
                for e in self.experiments:
                    out['instances'][b][m][e] = {}
                    for d in self.devices:
                        if self.no_docker:
                            temp[b][e][d]['docker'] = None
                        out['instances'][b][m][e][d] = temp[b][e][d]
        return out


parser = argparse.ArgumentParser()
parser.add_argument(
    '--device', type=str.lower,
    choices=ConfigManager.devices + ['all'],
    help='Run benchmarking for CPU or GPU or both', default='cpu')
parser.add_argument(
    '--backend', type=str.lower,
    choices=ConfigManager.backends + ['all'],
    help='Run benchmarking for tensorflow or pytoch or onnx', default='all')
parser.add_argument(
    '--count', type=int,
    help='How many iterations to take average from, for each experiment',
    default=50)
parser.add_argument(
    '--exp', nargs=argparse.REMAINDER, type=str.lower,
    default='all')
parser.add_argument(
    '--no-docker', action='store_true',
    help='Disable docker servers and try to connect servers running in the machine')

config = ConfigManager(parser.parse_args()).get_config_dict()

if __name__ == '__main__':
    print(config)
