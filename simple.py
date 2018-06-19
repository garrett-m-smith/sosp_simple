# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 08:57:55 2018

@author: garrettsmith

Beginning of a module for general-use SOSP simulations.

Will import dynamics functions from a dynamics module.
Will import plotting fns. from a plotting module.

Needs attributes of curr_state, state_hist, curr_harmony, harmony_hist,
centers, center_labels, n_dim, dim_names, max_sent_length, n_words,
prob. more.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from .dynamics import iterate, euclid_stop, vel_stop, cheb_stop, calc_harmony


class SimpleModel(object):
    def __init__(self, corpus_filename=None, centers=None,
                 local_harmonies=None, stopping_crit='euclid_stop'):
        assert corpus_filename is None or (centers is None
                                           and local_harmonies is None), \
            """Either specify a lexicon filename or centers and harmonies,
            but not both."""

        if corpus_filename is not None:
            self.read_corpus(corpus_filename)
        if centers is not None:
            assert len(centers.shape) > 1, 'centers must be a 2D array.'
            self.centers = centers
            self.n_dim = self.centers.shape[1]
        if local_harmonies is not None:
            assert len(local_harmonies) == self.centers.shape[0], \
             'Number of local harmonies does not match number of centers.'
            self.local_harmonies = local_harmonies
        if stopping_crit == 'vel_stop':
            self.stopping_crit = vel_stop
        elif stopping_crit == 'cheb_stop':
            self.stopping_crit = cheb_stop
        else:
            self.stopping_crit = euclid_stop

        self.tau = 0.01  # Time step for discretized dynamics
        self.max_time = 1000  # Max. number of time steps
        self._zero_state_hist()
        self.noise_mag = 0.001  # default
        self.gamma = 0.25
        self.tol = 0.05  # Stopping tolerance on each dim.

    def read_corpus(self, filename=None):
        print('Not yet implemented.')
        return True

    def _zero_state_hist(self):
        self.state_hist = np.zeros((self.max_time, self.n_dim))

    def set_noise_mag(self, noise_mag):
        self.noise_mag = noise_mag

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_tol(self, tol):
        self.tol = tol

    def set_max_time(self, max_time):
        self.max_time = max_time

    def set_local_harmonies(self, local_harmonies):
        self.local_harmonies = local_harmonies

    def single_run(self, init_cond=None):
        """Run the model once until stopping criterion is met or 
        time runs out.
        """
        assert init_cond is not None, 'Must set initial conditions.'
        assert len(init_cond) == self.state_hist.shape[1], \
            'Shape mismatch for initial conditions.'
        self.state_hist[0,] = init_cond
        noise = (np.sqrt(2 * self.noise_mag * self.tau)
                 * np.random.normal(0, 1, self.state_hist.shape))
        t = 0
        while t < self.max_time-1:
            if self.stopping_crit(self.state_hist[t], self.attrs, self.tol):
#            if self.stopping_crit(self.state_hist[t], self.centers, self.tol):
                self.state_hist[t+1,] = (self.state_hist[t,]
                        + self.tau * iterate(self.state_hist[t,], self.centers,
                                             self.local_harmonies, self.gamma)
                        + noise[t,])
                t += 1
            else:
                break

    def many_runs(self, n_runs=100, init_cond=None):
        """Do repeated Monte Carlo runs, potenially in different conditions.
        Returns a Pandas data frame with the center number and settling time.
        """
        print('Run number:')
        data_list = []
        for run in range(n_runs):
            if run % 100 == 0:
                print('[{}] '.format(run), end='')
            self._zero_state_hist()
            self.single_run(init_cond)
            trunc = self.state_hist[~np.all(self.state_hist == 0, axis=1)]
            for center in range(self.centers.shape[0]):
                if np.all(np.round(trunc[-1,]) == self.centers[center,]):
                    data_list.append([center, trunc.shape[0]])
        return pd.concat([pd.DataFrame([i], columns=('CenterNr', 'Time',))
                          for i in data_list])

    def run_multiple_conditions(self, n_runs=100, conditions=None,
                                init_cond=None):
        """Do many runs of multiple conditions by changing the harmony
        landscape between conditions.

        Assumes the harmonies for each center are a row of a NumPy array.
        """
        state_init = np.zeros((len(conditions), self.state_hist.shape[1]))
        if init_cond is not None:
            state_init = init_cond
        all_data = []
        for cond in range(conditions.shape[0]):
            self.locate_attrs()
            self.set_local_harmonies(conditions[cond,])
            print('Condition {}'.format(cond))
            cond_data = self.many_runs(n_runs, state_init[cond,])
            cond_data['Condition'] = [cond] * n_runs
            all_data.append(cond_data)
        return pd.concat(all_data)

    def neg_harmony(self, x, centers, local_harmonies, gamma):
        return -1 * calc_harmony(x, centers, local_harmonies, gamma)

    def jac_neg_harmony(self, x, centers, local_harmonies, gamma):
        return -1 * iterate(x, centers, local_harmonies, gamma)

    def locate_attrs(self):
        """Finds actual locations of attractors in the full harmony landscape
        using the Newton-CG algorithm on the negative of the harmony fn.
        """
        attrs = np.zeros(self.centers.shape)
        for c in range(self.centers.shape[0]):
            extremum = minimize(self.neg_harmony, self.centers[c],
                                args=(self.centers, self.local_harmonies,
                                      self.gamma), method='L-BFGS-B',
                                jac=self.jac_neg_harmony)
            attrs[c] = extremum.x
        unique_attrs = np.unique(np.round(attrs, 2), axis=0)
        self.attrs = unique_attrs
        print('Found {} unique attractors from {} centers'.format(
                self.attrs.shape[0], self.centers.shape[0]))
        return

    def plot_trace(self):
        trunc = self.state_hist[~np.all(self.state_hist == 0, axis=1)]
        plt.plot(trunc)
        plt.ylim(-0.01, 1.01)
        plt.xlabel('Time')
        plt.ylabel('Activation')
        plt.show()


if __name__ == '__main__':
    centers = np.array([[0., 1.], [1., 0.]])
    harmonies = np.array([1.0, 1.0])
    xinit = np.array([0., 0.])
    test = SimpleModel(None, centers, harmonies, 'cheb_stop')
    # test.set_tol(0.01)
    # test.single_run(xinit)
    # test.plot_trace()
    # results = test.many_runs(50, init_cond=xinit)
    # print(results.groupby('CenterNr').agg(['count', 'mean', 'std']))

    test.set_noise_mag(0.005)
    conds = np.array([[1.0, 1.0], [0.5, 1.0]])
    init = np.array([[0.5, 0.5], [0.5, 0.5]])
    many_results = test.run_multiple_conditions(n_runs=10, conditions=conds,
                                                init_cond=init)
    print(many_results.groupby(['Condition', 'CenterNr']).agg(
            ['count', 'mean', 'std']))
