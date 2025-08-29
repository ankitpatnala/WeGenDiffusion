import numpy as np
import matplotlib.pyplot as plt
import re

loss_log_paths = [('./results/DiT-B-2/train.log', 'unconditional'),
                  ('./results-month/DiT-B-2/train_org.log', 'month'),
                  ('/p/project1/training2533/lancelin1/WeGenDiffusion/results/DiT-B-2_season_1/train.log', 'season'),
                  ('/p/project1/training2533/lancelin1/WeGenDiffusion/results/DiT-B-2_previous_state/train.log', '-12h temperature')]
steps_per_epoch = 360 / 10
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
for loss_log_path, label in loss_log_paths:
    with open(loss_log_path, 'r') as f:
        raw_log = f.read()
    losses = [float(m) for m in loss_expr.findall(raw_log)]
    losses = np.array(losses)
    losses = smooth(losses, avg_window_size)

    x = np.arange(0, losses.shape[0]).astype(float)
    x /= steps_per_epoch

    plt.semilogy(x,losses, label=label)

plt.legend()
plt.xlabel ('epoch')
plt.ylabel('train loss')
plt.savefig('losses.png', dpi=300, bbox_inches='tight')
