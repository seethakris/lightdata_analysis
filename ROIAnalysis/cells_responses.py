import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import scipy.stats
import peakutils
from scipy.signal import detrend


class CorrelateReponses(object):
    def __init__(self, ExperimentFolder, ON_corr, OFF_corr, Both_corr, fastONtracelength, numberofplanes):
        self.numberofplanes = numberofplanes
        self.ResultFolder = os.path.join(ExperimentFolder, 'SaveResult')
        self.makedirs(self.ResultFolder)

        # Open saved files
        PickleFolder = os.path.join(ExperimentFolder, 'PickledData')
        self.Blue = self.open_pickled_files(PickleFolder, 'data_blue')
        self.Centroid = self.open_pickled_files(PickleFolder, 'centroid')

        ExperimentParameters = self.open_pickled_files(PickleFolder, 'experiment_parameters')
        self.stimulus_on_time = ExperimentParameters['stimulus_on_time']
        self.stimulus_off_time = ExperimentParameters['stimulus_off_time']
        self.framerate = ExperimentParameters['framerate']
        self.startframe = ExperimentParameters['frame_start']

        self.stimulus_on_time_ins = [np.round(ii / self.framerate) for ii in self.stimulus_on_time]
        self.stimulus_off_time_ins = [np.round(ii / self.framerate) for ii in self.stimulus_off_time]
        self.ITI = self.stimulus_off_time[0] - self.stimulus_on_time[0]

        self.fastONtracelength = fastONtracelength
        self.corr_threshold = {'fastON': ON_corr, 'slowON': ON_corr, 'OFF': OFF_corr, 'Both': Both_corr}
        self.colors = {'Both': 'yellow', 'fastON': 'blue', 'slowON': 'cyan', 'Inh': 'green', 'OFF': 'red'}

        self.dummytraces = self.dummy_trace()
        self.numcells = pd.DataFrame(columns=['Fish', 'Numcells', 'Plane', 'ResponseType', 'Type'])

        # Plot just plane data
        fs = plt.figure(figsize=(35, 15))
        gs = plt.GridSpec(6, numberofplanes, height_ratios=[2, 1, 1, 1, 2, 1])
        self.OnOffmatrix, self.OnOfftrace = self.run_through_data(fs, gs, self.Blue, 'Blue')
        plt.tight_layout()
        fs.savefig(os.path.join(self.ResultFolder, 'Cellfigure.pdf'), bbox_inches='tight')

        np.savez(os.path.join(self.ResultFolder, 'Cellresults.npz'), OnOffmatrix=self.OnOffmatrix,
                 OnOfftrace=self.OnOfftrace, numcells=self.numcells)

    def run_through_data(self, fig, grid, data, datatype):

        # Loop by plane
        for planes in xrange(1, self.numberofplanes):
            print planes
            OnOffmatrix = []
            data_perplane = []
            current_centroid = []
            for ii in xrange(0, np.size(data)):
                fish = data[ii]['fish']
                if fish.find('Plane' + str(planes)) > 0:

                    data_smooth = data[ii]['smooth_dff']
                    # data_smooth = detrend(data_smooth, axis=0)
                    print fish, np.shape(data_smooth)
                    # Find correct centroid for fish
                    for centroid in xrange(0, np.size(self.Centroid)):
                        if fish == self.Centroid[centroid]['fish']:
                            current_centroid = self.Centroid[centroid]
                            break

                    Correlationtrace = self.correlation_coefficient(fish, data_smooth)
                    self.plot_location(fig, grid, planes - 1, current_centroid, Correlationtrace)

                    OnOffmatrix.append(Correlationtrace)
                    data_perplane.append(data_smooth)

                    self.group_num_cells_by_plane(Correlationtrace, fish, planes)

            OnOfftrace = self.setup_for_plotting(fig, grid, planes - 1, data_perplane, OnOffmatrix,
                                                 title='Plane' + str(planes))
            ax1 = fig.add_subplot(grid[len(self.dummytraces) + 1, planes - 1])
            self.plot_traces(OnOfftrace)

        return OnOffmatrix, OnOfftrace

    def group_num_cells_by_plane(self, OnOffmatrix, fish, plane):
        for keys, values in OnOffmatrix.iteritems():
            numberofcells = (np.size(np.where(np.asarray(values) == 1)) / float(np.size(values, 0))) * 100
            # if keys.find('ON') >= 0:
            #     responsetype = 'ON'
            # elif keys.find('OFF') or keys.find('Both'):
            #     responsetype = 'OFF'
            # else:
            #     continue
            temp = pd.DataFrame([[fish, numberofcells, plane, keys, 'Cell']], columns=self.numcells.columns)
            self.numcells = self.numcells.append(temp, ignore_index=True)

    def correlation_coefficient(self, fish, data):
        # Correlate each cell with the dummy traces
        cells = np.size(data, 1)
        correlation_matrix = np.zeros((cells, len(self.dummytraces.keys())))
        for ii in xrange(0, cells):
            count = 0
            for types in self.dummytraces.itervalues():
                correlation_matrix[ii, count] = np.corrcoef(data[:, ii], types)[0, 1]
                count += 1

        dict_matrix = {}
        count = 0
        # print 'Fish Number: %s' % fish
        for types in self.dummytraces.keys():
            if types != 'Inh':
                dict_matrix[types] = (correlation_matrix[:, count] > self.corr_threshold[types])
            count += 1

        # Inhibition
        if 'Inh' in self.dummytraces:
            fastOn = self.dummytraces.keys().index('fastON')
            Inh = self.dummytraces.keys().index('Inh')
            correlation_list = []
            # Positive correlation wiht inhibition and negative corr with ON
            for ii in xrange(0, np.size(correlation_matrix, 0)):
                correlation_list.append(
                    correlation_matrix[ii, Inh] > 0.4 and correlation_matrix[ii, On] < -self.corr_threshold['fastON'])
            dict_matrix['Inh'] = np.asarray(correlation_list)
            for ii in xrange(0, np.size(dict_matrix['Inh'])):
                if dict_matrix['Inh'][ii] == True:
                    dict_matrix['OFF'][ii] = False

        for types in dict_matrix.iterkeys():
            print 'Number of %s cells is %s' % (types, np.size(np.where(dict_matrix[types] == 1)))
        print 'Total Number of %s cells' % (np.size(dict_matrix['fastON']))

        return dict_matrix

    def setup_for_plotting(self, fig, grid, columnnumber, data, Responsetrace, title):
        OnOffmatrix = self.combine_correlation_matrices_and_plot(fig=fig, grid=grid, columnnumber=columnnumber,
                                                                 data=data, Responsetrace=Responsetrace)
        return OnOffmatrix

    def combine_correlation_matrices_and_plot(self, fig, grid, columnnumber, data, Responsetrace):

        count = 0
        dict = {}
        for keys in Responsetrace[0].iterkeys():
            OnOff = np.array([])
            for ii in xrange(0, np.size(Responsetrace)):
                data_smooth = data[ii]
                temp = np.sort(data_smooth[:, Responsetrace[ii][keys]], axis=1)
                OnOff = np.hstack((OnOff, temp)) if OnOff.size else temp
            dict[keys] = OnOff

            # Plot
            with sns.axes_style('dark'):
                axis = fig.add_subplot(grid[count, columnnumber])
                cax = axis.imshow(OnOff.T, cmap='jet', interpolation='nearest', aspect='auto', vmin=-0.7,
                                  vmax=0.7)
                plt.grid('off')
                plt.title(keys)
                plt.xlabel('Time (seconds)')
                plt.ylabel('Cells')
                self.plot_stimulus_lines_onset(linewidth=1)
                plt.locator_params(axis='y', nbins=6)
                plt.xlim([0, np.size(OnOff, 0)])
                plt.ylim([np.size(OnOff, 1) - 1, 0])
                fig.colorbar(cax, ticks=[-0.2, 0, 0.2])
                count += 1
        return dict

    def plot_location(self, fig, grid, columnnumber, centroid, Responses):
        x = centroid['x']
        y = centroid['y']

        ax1 = fig.add_subplot(grid[len(self.dummytraces), columnnumber])
        count = 0
        plt.plot(x, y, 'o', markersize=10, markeredgewidth=1, markeredgecolor='k', markerfacecolor='None', alpha=0.5)
        for keys, values in Responses.iteritems():
            plt.plot(x[values], y[values], 'o', markersize=10, markeredgewidth=1, markeredgecolor=self.colors[keys],
                     markerfacecolor=self.colors[keys], label=keys, alpha=0.5)
            count += 1
            # plt.axis('off')
        plt.xlim((180, 0))
        plt.ylim((0, 90))
            # plt.gca().invert_yaxis()

    def plot_traces(self, OnOfftrace):
        count = 0
        for keys, values in OnOfftrace.iteritems():
            if keys != 'NoResponse':
                # Mean and confidence interval
                templist = values * 100
                label_in_seconds = self.convert_frames_to_seconds(np.size(templist, 0))[:-1]
                y, error1, error2 = self.mean_confidence_interval(templist)
                # print np.shape(label_in_seconds), np.shape(y), np.shape(error1)
                plt.plot(label_in_seconds, y, label=keys, linewidth=2, alpha=0.8, color=self.colors[keys])
                count += 1
                # plt.fill_between(label_in_seconds, error1, error2, alpha=0.5)
        self.plot_stimulus_lines_onset(secondsflag=True)
        plt.axhline(y=0, linestyle='-', color='k', linewidth=0.5)
        plt.grid('off')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Delta f/f')
        plt.legend()

    def dummy_trace(self):

        # Create dummy traces for correlation
        time = np.size(self.Blue[0]['smooth_dff'], 0)
        slowOn_trace = np.zeros(time)
        fastOn_trace = np.zeros(time)
        Off_trace = np.zeros(time)
        On_and_off_trace = np.zeros(time)

        for ii in xrange(0, len(self.stimulus_on_time)):

            slowOn_trace[self.stimulus_on_time[ii] + self.fastONtracelength:self.stimulus_off_time[ii]] = 1
            fastOn_trace[self.stimulus_on_time[ii]:self.stimulus_on_time[ii] + self.fastONtracelength] = 1
            On_and_off_trace[self.stimulus_on_time[ii]:self.stimulus_on_time[ii] + self.fastONtracelength] = 1
            On_and_off_trace[self.stimulus_off_time[ii]:self.stimulus_off_time[ii] + self.fastONtracelength] = 1

            if ii == len(self.stimulus_on_time) - 1:
                Off_trace[self.stimulus_off_time[ii]:self.stimulus_off_time[ii] + (self.ITI / 2)] = 1

            else:
                Off_trace[self.stimulus_off_time[ii]:self.stimulus_on_time[ii + 1] - (self.ITI / 2)] = 1
        return {'slowON': slowOn_trace, 'fastON': fastOn_trace, 'OFF': Off_trace, 'Both': On_and_off_trace}

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a, 1), scipy.stats.sem(a, 1)
        h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h

    def convert_frames_to_seconds(self, n):
        ## Convert frames to seconds
        time = 1.0 / self.framerate
        label_in_seconds = np.linspace(0, n * time, n + 1)

        return label_in_seconds

    @staticmethod
    def makedirs(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    @staticmethod
    def open_pickled_files(Folder, filename):
        with open(os.path.join(Folder, filename)) as fp:
            return pickle.load(fp)

    def plot_stimulus_lines_onset(self, linewidth=2, secondsflag=False):
        if secondsflag:
            for ii in xrange(0, np.size(self.stimulus_on_time_ins)):
                plt.axvline(x=self.stimulus_on_time_ins[ii], linestyle='-', color='k', linewidth=linewidth)
                plt.axvline(x=self.stimulus_off_time_ins[ii], linestyle='--', color='k', linewidth=linewidth)
        else:
            for ii in xrange(0, np.size(self.stimulus_on_time)):
                plt.axvline(x=self.stimulus_on_time[ii], linestyle='-', color='k', linewidth=linewidth)
                plt.axvline(x=self.stimulus_off_time[ii], linestyle='--', color='k', linewidth=linewidth)


if __name__ == '__main__':
    Folder = '/Users/seetha/Desktop/Revised_manuscript/Thalamus/ROI_Thalamus_1020Gcamp6_1fps/'
    ON_corr = 0.5
    OFF_corr = 0.5
    numberofplanes = 1

    CorrelateReponses(Folder, ON_corr, OFF_corr, numberofplanes)
