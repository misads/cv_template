import os

commands = {
    'cascaded': 'python3 eval.py --tag Cascaded --model cascaded --load checkpoints/Cascaded --which-epoch 669',
    'pure3': 'python3 eval.py --tag pure3 --model default --load checkpoints/pure3 --which-epoch 499',

}


def eval(which):
    os.system(commands[which])


if __name__ == '__main__':
    eval('pure3')
    # eval('cascaded')
