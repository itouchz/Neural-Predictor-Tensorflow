import os

from nasbench import api
from proxylessnas.backbone import SuperProxylessNAS
from nasbench_nlp.nas_environment import Environment

def load_nasbench_101():
    if not os.path.exists('./nasbench_only108.tfrecord'):
        os.system('curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord')

    return api.NASBench('./nasbench_only108.tfrecord')


def load_nasbench_nlp():
    precomputed_logs_path = 'nasbench_nlp/train_logs_single_run/'
    
    env = Environment(precomputed_logs_path)
    search_set = env.get_precomputed_recepies()
    return search_set, env

def load_proxylessnas():
    return SuperProxylessNAS