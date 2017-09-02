""" Run Regression On Data """

__author__ = "Seetha Krishnan"
__copyright__ = "Copyright (C) 2016 Seetha Krishnan"
__license__ = "Public Domain"
__version__ = "1.0"


from numpy import r_, hanning, convolve, zeros, asarray, shape, size, mean, arange
from thunder import RegressionModel
from thunder import Colorize
from copy import copy
from matplotlib.colors import ListedColormap
import palettable
import matplotlib.pyplot as plt
import seaborn as sns
import math
import collections
from scipy.stats.mstats import zscore
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib as mpl


class class_regression(object):
    def __init__(self, data, img_raw, time_experiment, stimulus_on_time, stimulus_off_time, stimulus_train):
        self.data = data
        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.stimulus_train = stimulus_train
        self.image = Colorize.image
        self.time_experiment = time_experiment
        self.parameters = ['Onset', 'xOffset']
        self.reference = mean(img_raw, 0)  # Get mean_image

        # Get unique stimuli and set up labels for regressors
        self.unique_stimuli = list(set(self.stimulus_train))
        self.regression_types = [ii + '_' + jj for ii in self.unique_stimuli for jj in self.parameters]
        self.dict_each_stim = [ii + '_' + jj for ii in self.stimulus_train for jj in self.parameters]

        print 'Creating regressors for the following types : ', self.regression_types

    def create_regression_parameters(self, fig1, amplitude_for_stimulus_train, smooth_window_length, plot_flag=0):

        # create stimulus traces with given stimuli parameters
        stim_trace = self.create_stimulus_traces(amplitude_for_stimulus_train)

        # Smooth using hanning window to mimic calcium dynamics
        stim_trace_smoothed = zeros(shape(stim_trace))

        for ii in xrange(0, size(stim_trace, 1)):
            stim_trace_smoothed[:, ii] = self.smooth_hanning(stim_trace[:, ii], smooth_window_length)

        if plot_flag:
            self.plot_regressor(fig1, stim_trace_smoothed)

        # Order regressors in alphabetical order
        regressors = {}
        for ii in xrange(0, len(self.regression_types)):
            regressors[self.regression_types[ii]] = stim_trace_smoothed[:, ii]

        ordered_regressors = collections.OrderedDict(sorted(regressors.items()))

        # Get list of regressors for performing regression
        regressorlist = list()
        for key, value in ordered_regressors.iteritems():
            temp = value
            regressorlist.append(temp)

        return ordered_regressors, regressorlist

    def create_stimulus_traces(self, amplitude_for_stimulus_train):
        # loop through stimulus on and off times to create regressors
        stim_traces = zeros((self.time_experiment, len(self.regression_types)))
        stimulus_times_combined = asarray((self.stimulus_on_time, self.stimulus_off_time)).T.flatten()

        for ii in xrange(0, len(self.regression_types)):
            for jj in xrange(0, len(self.dict_each_stim) - 1, 2):
                if self.regression_types[ii] == self.dict_each_stim[jj]:
                    stim_traces[stimulus_times_combined[jj]:stimulus_times_combined[jj + 1] - 5,
                    ii] = amplitude_for_stimulus_train

                    if jj != len(self.dict_each_stim) - 2:
                        stim_traces[stimulus_times_combined[jj + 1]:stimulus_times_combined[jj + 2] - 5,
                        ii + 1] = amplitude_for_stimulus_train
                    else:
                        stim_traces[stimulus_times_combined[jj + 1]:-5, ii + 1] = amplitude_for_stimulus_train

        return stim_traces

    def plot_regressor(self, fig1, stim_traces):

        for ii in xrange(0, len(self.unique_stimuli)):
            ax1 = fig1.add_subplot(len(self.unique_stimuli), 1, ii + 1)
            for jj in xrange(0, len(self.regression_types)):
                if self.unique_stimuli[ii] in self.regression_types[jj]:
                    plt.plot(stim_traces[:, jj], label=self.regression_types[jj])

            ax1.legend(prop={'size': 14}, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True,
                       shadow=True)
            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            self.plot_stimulus_patch(ax1)

            if ii == 0:
                plt.title('Regressors')

    def plot_correlation_with_regressors(self, fig1, gs, regressor_key, regressor_value, gridspecs='[0,0]',
                                         color_bar=False, region='Habenula', clim=[-1, 1]):

        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        corrs = self.data.correlate(regressor_value)
        corrMat = self.center(corrs.pack())
        ax1 = self.image(corrMat, cmap='seismic')

        plt.title(regressor_key + ' : ' + region, fontsize=12)

        if color_bar:
            c_axis = eval('fig1.add_subplot(gs[' + str(len(self.regression_types)) + ', 1])')
            plt.colorbar(ax1, cax=c_axis, ticks=[-1, 0, 1], orientation='horizontal')

        plt.tight_layout()

        return corrMat

    def perform_regression(self, regressor_list):
        regressor_list = zscore(regressor_list, axis=1)
        regression_results = RegressionModel.load(regressor_list, 'linear').fit(self.data)
        b = regression_results.select('betas').pack()
        rsq = regression_results.select('stats').pack()

        return regression_results, b, rsq

    def plotimageplanes(self, fig1, gs, img, cmap='gray', color_bar=False, plot_title='Habenula', gridspecs='[0, 0]',
                        **kwargs):

        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        plt.title(plot_title, fontsize=14)
        if 'clim' in kwargs:
            h = self.image(img, cmap=cmap, clim=kwargs.values()[0])
        else:
            h = self.image(img, cmap=cmap)

        if color_bar:
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(h, cax=cax)
            tick_locator = ticker.MaxNLocator(nbins=3)
            cb.locator = tick_locator
            cb.update_ticks()

    def plot_regressors_as_RGB(self, fig1, gs, regressors, rsq, b, colors, brightness_scale=3, mixing_parameter=0.5):

        b_pos = b * (b > 0)  # Get positive beta values
        # Set up colorize function
        c = Colorize(cmap='indexed', scale=brightness_scale, flag_scale=1)

        for ii in xrange(0, len(regressors)):
            ax1 = fig1.add_subplot(gs[0, ii])
            c.colors = [colors[ii]]
            img = c.transform([b_pos[ii, :, :]], mask=rsq, background=self.reference, mixing=mixing_parameter)
            self.image(img)
            plt.title(regressors.keys()[ii])

        # Plot the unique stimuli in subplots
        subplot_count = 0
        for ii in self.unique_stimuli:
            b_pos_list = []
            color_mat = []
            for keys in regressors.iterkeys():
                if ii in keys:
                    index = regressors.keys().index(keys)
                    b_pos_list.append(b_pos[index, :, :])
                    color_mat.append(colors[index])
            ax1 = fig1.add_subplot(gs[1, subplot_count])
            c.colors = color_mat
            img = c.transform(b_pos_list, mask=rsq, background=self.reference, mixing=mixing_parameter)
            plt.title(ii)
            self.image(img)
            subplot_count += 1

        # Plot the different stimuli parameters in subplots
        for ii in self.parameters:
            b_pos_list = []
            color_mat = []
            for keys in regressors.iterkeys():
                if ii in keys:
                    index = regressors.keys().index(keys)
                    b_pos_list.append(b_pos[index, :, :])
                    color_mat.append(colors[index])
            ax1 = fig1.add_subplot(gs[1, subplot_count])
            c.colors = color_mat
            img = c.transform(b_pos_list, mask=rsq, background=self.reference, mixing=mixing_parameter)
            plt.title(ii)
            self.image(img)
            subplot_count += 1

        # Plot all together
        b_pos_list = []
        color_mat = []
        for ii in xrange(0, len(regressors)):
            b_pos_list.append(b_pos[ii, :, :])
            color_mat.append(colors[ii])
        ax1 = fig1.add_subplot(gs[2:4, 1:len(regressors)-1])
        c.colors = color_mat
        img = c.transform(b_pos_list, mask=rsq, background=self.reference, mixing=mixing_parameter)
        self.image(img)
        subplot_count += 1

        ax1 = fig1.add_subplot(gs[2, 0])
        self.create_colorbar(ax1, regressors, colors)

    def plot_all_together_inseperateplot(self, fig1, gs, regressors, rsq, b, colors, brightness_scale=3,
                                         mixing_parameter=0.8, gridspecs='[0, 0]'):

        b_pos = b * (b > 0)  # Get positive beta values
        # Set up colorize function
        c = Colorize(cmap='indexed', scale=brightness_scale, flag_scale=1)

        b_pos_list = []
        color_mat = []
        for ii in xrange(0, len(regressors)):
            b_pos_list.append(b_pos[ii, :, :])
            color_mat.append(colors[ii])
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        c.colors = color_mat
        img = c.transform(b_pos_list, mask=rsq, background=self.reference, mixing=mixing_parameter)
        self.image(img)

    @staticmethod
    def center(m):
        y = m.copy()
        y[y > 0] = y[y > 0] / max(y[y > 0])
        y[y < 0] = y[y < 0] / -min(y[y < 0])
        return y

    @staticmethod
    def smooth_hanning(x, window_len):
        s = r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        w = hanning(window_len)
        y = convolve(w / w.sum(), s, mode='valid')

        return y[:-window_len + 1]

    @staticmethod
    def create_colorbar(ax1, regressors, colors):

        labels = [keys.replace('_', ' ') for keys in regressors.iterkeys()]
        cmap = mpl.colors.ListedColormap(colors)

        bounds = arange(0.5, len(regressors) + 1, 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        cb2 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=range(1, len(regressors) + 1),
                                        orientation='horizontal',
                                        drawedges=True)

        cb2.ax.set_xticklabels(labels, rotation=90, fontsize=15)

    def plot_vertical_lines_onset(self):
        for ii in xrange(0, size(self.stimulus_on_time)):
            plt.axvline(x=self.stimulus_on_time[ii], linestyle='-', color='k', linewidth=2)

    def plot_vertical_lines_offset(self):
        for ii in xrange(0, size(self.stimulus_off_time)):
            plt.axvline(x=self.stimulus_off_time[ii], linestyle='--', color='k', linewidth=2)

    def plot_stimulus_patch(self, ax1):

        # Get y grid size
        y_tick = ax1.get_yticks()
        y_tick_width = y_tick[2] - y_tick[1]

        # Adjust axis to draw patch
        y_lim = ax1.get_ylim()
        ax1.set_ylim((y_lim[0], y_lim[1] + y_tick_width))

        for ii in xrange(0, size(self.stimulus_on_time)):
            # Find time of stimulus for width of patch
            time_stim = self.stimulus_off_time[ii] - self.stimulus_on_time[ii]

            # Check different cases of stimuli to create patches
            if self.stimulus_train[ii] == "Blue":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim,  y_tick_width/2, fc='b')
                ax1.add_patch(rectangle)
            elif self.stimulus_train[ii] == "UV":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim,  y_tick_width/2, fc='aqua')
                ax1.add_patch(rectangle)
            elif self.stimulus_train[ii] == "US":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim,  y_tick_width/2, fc='orange')
                ax1.add_patch(rectangle)
            elif self.stimulus_train[ii] == "Red":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim,  y_tick_width/2, fc='red')
                ax1.add_patch(rectangle)