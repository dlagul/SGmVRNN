import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib as mpl

class Logger():
    def __init__(self, out, name='loss', xlabel='epoch'):
        self.out = out
        self.name = name
        self.xlabel = xlabel
        self.txt_file = os.path.join(out, name + '.txt')
        self.plot_file = os.path.join(out, name + '.png')
        self.plot_kl_file = os.path.join(out, name + '_kl.png')

    def log_trainer(self, epoch, states, t=None):
        self._print_trainer(epoch, states, t)
        self._plot(epoch, states)
        states_kl = {}
        states_kl['Epoch'] = states['Epoch']
        states_kl['KL_pi'] = states['KL_pi']
        states_kl['KL_z'] = states['KL_z']
        self._plot_kl(epoch, states_kl)  

    def log_tester(self, epoch, states, t=None):
        self._print_tester(epoch, states, t)

    def log_evaluator_pot(self, message):
        self._print_eval_result(message)

    def _print_trainer(self, epoch, states, t=None):
        if t is not None:
            if self.xlabel == 'epoch':
                message = '(eps: %d, time: %.5f) ' % (epoch, t)
            else:
                message = '(%s: %d, time: %.5f) ' % (self.xlabel, epoch, t)
        else:
            if self.xlabel == 'epoch':
                message = '(eps: %d) ' % (epoch)
            else:
                message = '(%s: %d) ' % (self.xlabel, epoch)
        for k, v in states.items():
            message += '%s: %.5f ' % (k, v)

        with open(self.txt_file, "a") as f:
            f.write('%s\n' % message)
    
    def _print_tester(self, epoch, states, t=None):
        
        message = '{},{},{}'.format(states['Last_timestamp'],
                                    states['Llh_Lt'],
                                    states['IA'])
        with open(self.txt_file, "a") as f:
            f.write('%s\n' % message)
    
    def _print_eval_result(self, message):
        with open(self.txt_file, "a") as f:
            f.write('%s\n' % message)
    
    def _plot(self, epoch, states):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(states.keys())}
        self.plot_data['X'].append(epoch)
        self.plot_data['Y'].append(
            [states[k] for k in self.plot_data['legend']])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()
        for i, k in enumerate(self.plot_data['legend']):
            ax.plot(np.array(self.plot_data['X']),
                    np.array(self.plot_data['Y'])[:, i],
                    label=k)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.name)
        l = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.savefig(self.plot_file,
                    bbox_extra_artists=(l, ),
                    bbox_inches='tight')
        plt.close()

    def _plot_kl(self, epoch, states):
        if not hasattr(self, 'plot_kl'):
            self.plot_kl = {'X': [], 'Y': [], 'legend': list(states.keys())}
        self.plot_kl['X'].append(epoch)
        self.plot_kl['Y'].append(
            [states[k] for k in self.plot_kl['legend']])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid()
        for i, k in enumerate(self.plot_kl['legend']):
            ax.plot(np.array(self.plot_kl['X']),
                    np.array(self.plot_kl['Y'])[:, i],
                    label=k)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.name)
        l = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.savefig(self.plot_kl_file,
                    bbox_extra_artists=(l, ),
                    bbox_inches='tight')
        plt.close()

