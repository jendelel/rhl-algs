import numpy as np


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def seconds_to_text(secs):
    # From: https://stackoverflow.com/questions/4048651/python-function-to-convert-seconds-into-minutes-hours-and-days/4048773
    days = secs // 86400
    hours = (secs - days * 86400) // 3600
    minutes = (secs - days * 86400 - hours * 3600) // 60
    seconds = int(secs - days * 86400 - hours * 3600 - minutes * 60)
    result = ("{} days, ".format(days) if days else "") + \
             ("{} h, ".format(hours) if hours else "") + \
             ("{} m, ".format(minutes) if minutes else "") + \
             ("{} s".format(seconds))
    return result
