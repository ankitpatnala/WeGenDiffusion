import numpy as np
import matplotlib.pyplot as plt
import re

loss_log_paths = ['./results/DiT-B-2/train.log',
                  './results-month/DiT-B-2/train.log']
avg_window_size = 256

colors=['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
        '#AA3377', '#BBBBBB', '#000000']
default_cycler = plt.cycler(color=colors)
plt.rc('axes', prop_cycle=default_cycler)

def smooth(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

loss_expr = re.compile(r'Loss = (\d.\d\d\d\d)')
for loss_log_path in loss_log_paths:
    with open(loss_log_path, 'r') as f:
        raw_log = f.read()
    losses = [float(m) for m in loss_expr.findall(raw_log)]
    losses = np.array(losses)
    losses = smooth(losses, avg_window_size)

    plt.semilogy(losses, label=loss_log_path)

plt.legend()
plt.savefig('losses.png')
