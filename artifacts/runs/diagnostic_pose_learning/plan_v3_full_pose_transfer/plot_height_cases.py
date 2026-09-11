"""Plot saved MuJoCo low/high traces; this does not run physics."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
colors = {'initial': '#777777', 'final': '#1565C0'}
for col, case in enumerate(('low_stand', 'high_stand')):
    for phase in ('initial', 'final'):
        with np.load(root / f'{phase}_mujoco_dev8' / case / 'trace.npz', allow_pickle=False) as trace:
            time = trace['times']
            color = colors[phase]
            axes[0, col].plot(time, trace['errors'], color=color, label=phase, linewidth=1.4)
            axes[1, col].plot(time, trace['orientation_errors_rad'], color=color, linewidth=1.4)
            axes[2, col].plot(time, trace['base'][:, 2], color=color, linewidth=1.4)
            names = trace['contact_body_names'].astype(str)
            feet = np.array([name.endswith('_foot') for name in names])
            assert feet.sum() == 4, names
            count = (trace['contacts'][:, feet] > 1).sum(axis=1)
            axes[3, col].plot(time, count, color=color, linewidth=1.1)
            if time[-1] < 59.99:
                for ax in axes[:, col]:
                    ax.axvline(time[-1], color=color, linestyle=':', linewidth=1)
    axes[0, col].set_title(case.replace('_', ' ').title())
    axes[0, col].legend(loc='upper left', frameon=False)
    axes[0, col].set_ylim(bottom=0)
    axes[1, col].set_ylim(bottom=0)
    axes[2, col].set_ylim(.1, .4)
    axes[3, col].set_ylim(-.2, 4.3)
    axes[3, col].set_yticks([0, 1, 2, 3, 4])
    axes[3, col].set_xlabel('Saved state time (s)')
    for ax in axes[:, col]:
        ax.axvspan(20, 60, color='#E8F1F8', alpha=.6, zorder=-1)
        ax.set_xlim(0, 60)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
for ax, label in zip(axes[:, 0], ('TCP position error (m)', 'TCP orientation error (rad)', 'Base height (m)', 'Feet with net force > 1 N')):
    ax.set_ylabel(label)
fig.suptitle('Full-pose transfer: local low-target gain, high-target failure', fontsize=14, y=.995)
fig.text(.5, .955, 'Neutral500 initialization vs 500 transfer iterations | MuJoCo saved trajectories', ha='center', fontsize=10)
fig.text(.07, .012, 'Blue shading: requested 20–60 s hold. Dotted line: initial high case ended at 20.04 s.\nInitial low uses nonfoot support; final low/high do not in the hold. Foot counts alone do not establish task success.\nSaved contact net-force samples; provisional B-frame experiment. Render with the pawweaver-data environment.', fontsize=8.5)
fig.tight_layout(rect=(0, .06, 1, .94))
fig.savefig(root / 'low_high_saved_trace.png', dpi=160)
