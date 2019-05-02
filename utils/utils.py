import numpy as np
import datetime
import os


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_log_dir(args):
    return os.path.join(
            "runs", "experiments", "{}-{}-{}".format(args.env, args.alg,
                                                     datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))


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
