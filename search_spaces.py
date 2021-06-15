import os

from nasbench import api
from proxylessnas.backbone import SuperProxylessNAS

def load_nasbench_101():
    if not os.path.exists('nasbench_only108.tfrecord'):
        os.system('curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord')

    return api.NASBench('nasbench_only108.tfrecord')


def NAS_Bench_201():
    return data

def load_proxylessnas():
    return SuperProxylessNAS