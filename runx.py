# encoding = utf-8
"""
    借鉴自NVIDIA的深度学习实验管理工具runx(https://github.com/NVIDIA/runx)。自己重写了一下。
    可以按照yml配置文件实现grid_searching。

    用法：
    python runx.py --sweep sweep.yml --show  # 默认

    python runx.py --sweep sweep.yml --run


"""
import argparse
import misc_utils as utils
import random
import string
import yaml
import os


def load_yml(file='sweep.yml', op=None):
    if not os.path.isfile(file):
        raise FileNotFoundError('File "%s" not found' % file)

    with open(file, 'r') as f:
        try:
            cfg = yaml.safe_load(f.read())
        except yaml.YAMLError:
            raise Exception('Error parsing YAML file: ' + file)

    if op:
        return cfg[op]
    else:
        return cfg


def hash(n):
    choices = '0123456789abcdef'
    ans = ''
    for _ in range(n):
        ans += choices[random.randint(0, 15)]

    return ans


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--sweep', type=str, default='sweep.yml', help='configure file, default is "sweep.yml"')
    parser.add_argument('--show', action='store_true', default=True)
    parser.add_argument('--run', action='store_true')

    return parser.parse_args()


opt = parse_args()


if __name__ == '__main__':

    cfg = load_yml(opt.sweep)

    cmd = cfg['cmd']

    hparams = cfg['hparams'].items()
    hparams = list(hparams)

    n = len(hparams)

    temp = [''] * n

    ans = []

    def dfs(i):
        if i >= n:
            ans.append(temp.copy())
            # print(temp)
            return
        hparam, choices = hparams[i]

        for choice in choices:
            temp[i] = choice
            dfs(i+1)

    dfs(0)
    # for hparam in hparams:
    for i, one_run in enumerate(ans):
        command = cmd + ' --tag %s' % hash(8)
        for (hparam, _), choice in zip(hparams, one_run):
            command += ' --%s %s' % (hparam, choice)

        utils.color_print(('%d/%d: ' % ((i+1), len(ans)) + command), 4)
        if opt.run:
            os.system(command)



