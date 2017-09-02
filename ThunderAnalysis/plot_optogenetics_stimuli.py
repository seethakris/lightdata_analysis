""" Analyse and plot results from optogenetic stimulation """

__author__ = "Seetha Krishnan"
__copyright__ = "Copyright (C) 2016 Seetha Krishnan"
__license__ = "Public Domain"
__version__ = "1.0"


from numpy import r_, ones, convolve
import pandas as pd
import os
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set_context('paper', font_scale=1.8)


class CollectandPlot(object):
    def __init__(self, FileName, pulse_time):
        self.FileName = FileName
        self.pulse_time = pulse_time


        # Load and convert to pandas dataframe
        if os.path.exists(self.FileName[:-4] + '.csv'):
            self.dataframe = pd.read_csv(self.FileName[:-4] + '.csv', sep='\t')
        else:
            os.rename(self.FileName, self.FileName[:-4] + '.csv')
            self.dataframe = pd.read_csv(self.FileName[:-4] + '.csv', sep='\t')

    def process_data(self):
        data = self.dataframe.values[:, 1:-2]
        zscore_data = (data - np.mean(data, 0)) / (np.std(data, 0) + 0.001)

        time_drop = np.where(zscore_data < -1.5)[0]
        cell_drop = np.where(zscore_data < -1.5)[1]
        zscore_data[time_drop, cell_drop] = 0

        smooth_data = np.zeros(np.shape(zscore_data))
        for ii in xrange(0, np.size(zscore_data, 0)):
            smooth_data[ii, :] = self.smooth_func(zscore_data[ii, :], 5)

        mean_data = np.mean(smooth_data, 1)
        error_data = 1.96 * sem(smooth_data, 1)

        return smooth_data, mean_data, error_data

    def plot_data(self, fig1, gs, data, mean_data, legend_flag, title, vline=1, gridspecs='[0,0]'):
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')

        with sns.axes_style('dark'):
            plt.plot([], sns.xkcd_rgb["denim blue"], label='cell traces')
            plt.plot(data, sns.xkcd_rgb["denim blue"], alpha=0.5)
            plt.plot(mean_data, sns.xkcd_rgb["pale red"], linewidth=2, label='mean')
            if vline:
                plt.axvline(x=self.pulse_time, linestyle='--', color='k', linewidth=2, label='stimulation')

            plt.xlim([0, np.size(data, 0)])
            plt.ylim([-2, 4])

            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax1.set(xlabel='Time (seconds)', ylabel='Z-score')

            plt.title(title, fontsize=12)
            plt.grid('off')

            plt.locator_params(axis='y', nbins=4)
            if legend_flag:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    def calculate_response_stats(self, data):

        before_amp = np.mean(np.max(data[0:self.pulse_time, :], 0))
        after_amp = np.mean(np.max(data[self.pulse_time:, :], 0))

        bef_criteria = np.max(data[0:self.pulse_time, :], 0)
        aft_criteria = np.max(data[self.pulse_time:self.pulse_time + 10, :], 0)
        cell_number = (np.where((bef_criteria < aft_criteria)) and np.where((aft_criteria > 1.5)))[0]

        print cell_number

        num_cells_with_response = np.size(cell_number)

        perc_cells_with_response = (num_cells_with_response / (np.float(np.size(data, 1)))) * 100

        return before_amp, after_amp, perc_cells_with_response

    def plot_stats(self, fig1, gs, dataframe):
        with sns.axes_style('darkgrid'):
            bef = dataframe['Amplitude Before Stimulation']
            aft = dataframe['Amplitude After Stimulation']
            cells = dataframe['Percentage of active cells']
            # cmapCat = ListedColormap(sns.color_palette("Paired", n_colors=np.size(bef)), name='from_list').colors

            x = np.ones(np.size(bef))

            ax1 = eval('fig1.add_subplot(gs' + '[0,0]' + ')')

            ax1.plot([x * 1, x * 2], [bef, aft], 's-', markersize=8, markeredgecolor='black', markeredgewidth=1)
            plt.locator_params(axis='y', nbins=5)
            plt.xlim([0, 3])
            plt.ylim([0, 2.5])
            plt.xticks([1, 2], ['Before', 'After'], rotation='vertical')
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

            ax2 = eval('fig1.add_subplot(gs' + '[0,2]' + ')')
            ax2.plot(x * 1, cells, 's', markersize=8, markeredgecolor='black', markeredgewidth=1, alpha=0.5)
            ax2.set(ylabel='% of responding cells')
            plt.xlim([0, 2])
            plt.ylim([0, 60])

    @staticmethod
    def smooth_func(x, window_len=10):
        s = r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        w = ones(window_len, 'd')
        y = convolve(w / w.sum(), s, mode='valid')
        return y[window_len / 2:-window_len / 2 + 1]

    def plot_vertical_lines_onset(self):
        for ii in xrange(0, np.size(self.stimulus_on_time)):
            plt.axvline(x=self.stimulus_on_time[ii], linestyle='-', color='k', linewidth=2)

    def plot_vertical_lines_offset(self):
        for ii in xrange(0, np.size(self.stimulus_off_time)):
            plt.axvline(x=self.stimulus_off_time[ii], linestyle='--', color='k', linewidth=2)
