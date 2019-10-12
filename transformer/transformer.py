import numpy as np


def get_angles(pos, i, d_model):
    """
    :param pos: word position
    :param i:在词向量中的位置
    :param d_model: dimension of embedding
    :return:
    """
    angle_rates = 1 / (np.power(10000, (2 * (i // 2)) / d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):


