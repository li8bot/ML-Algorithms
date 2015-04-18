# coding:utf-8


def check_data(data):
    if not isinstance(data, list):
        try:
            data = list(data)
        except TypeError:
            return False

    if not all(isinstance(item, int) or isinstance(item, float) for item in data):
        return False

    return data


def mode(data):
    data = check_data(data)

    if not data:
        return None

    if len(data) == 0:
        return None

    m = max([data.count(a) for a in data])

    if m > 1:
        for val in sorted(data):
            if data.count(val) == m:
                return val
    else:
        return min(data)


def median(data):
    data = check_data(data)

    if not data:
        return None

    data = sorted(data)

    if len(data) == 0:
        return None

    if len(data) % 2 == 1:
        return data[((len(data) + 1) / 2) - 1]

    if len(data) % 2 == 0:
        return float(sum(data[(len(data) / 2) - 1:(len(data) / 2) + 1])) / 2.0


def mean(data):
    data = check_data(data)

    if not data:
        return None

    return sum(data)/len(data)