# encoding = utf-8


class Meters(object):
    def __init__(self):
        pass

    def update(self, new_dic):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def items(self):
        return self.dic.items()


class AverageMeters(Meters):
    """AverageMeter class

    Example
        >>> avg_meters = AverageMeters()
        >>> for i in range(100):
        >>>     avg_meters.update({'f': i})
        >>>     print(str(avg_meters))

    """

    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


class ExponentialMovingAverage(Meters):
    """EMA class

    Example
        >>> ema_meters = ExponentialMovingAverage(0.98)
        >>> for i in range(100):
        >>>     ema_meters.update({'f': i})
        >>>     print(str(ema_meters))

    """

    def __init__(self, decay=0.9, dic=None, total_num=None):
        self.decay = decay
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        decay = self.decay
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = (1 - decay) * new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] = decay * self.dic[key] + (1 - decay) * new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key]  # / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()