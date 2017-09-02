import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from collections import Counter
from scipy.signal import detrend
from scipy.stats import sem
from Tools import PlottingTools
from scipy.cluster.vq import kmeans, vq

sns.set_context('paper', font_scale=1.2)
filesep = os.path.sep


class Get_Compile_Data(object):
    def __init__(self, WorkingDirectory, Result_Directory, Figure_Directory, stimulus_on_time, stimulus_off_time,
                 stimulus_types, baseline, smooth_window):

        self.WorkingDirectory = WorkingDirectory
        self.Result_Directory = Result_Directory
        self.Figure_Directory = Figure_Directory
        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.stimulus_types = stimulus_types
        self.smooth_window = smooth_window
        self.baseline = baseline

    def LoadData(self):
        mat_filenames = [f for f in os.listdir(self.WorkingDirectory) if f.endswith('.mat')]
        pp = PdfPages(self.Figure_Directory + filesep + 'Experiment_stats.pdf')

        calcium_data_allfish = np.array([])

        for ii in mat_filenames:
            print 'Collecting File...', ii
            # Load Data
            alldata = scipy.io.loadmat(os.path.join(self.WorkingDirectory, ii))

            # Preprocess data
            row, x, y, calcium = self.remove_neuropils(alldata)
            Fish_Name = int(ii[ii.find('Fish') + 4:ii.find('Fish') + 7])
            fish = np.repeat(Fish_Name, np.size(x))
            self.plot_mean_std_historgram(calcium, title_figure='Raw Data', pdf=pp)

            calcium_detrend = self.detrend_calcium_trace(calcium)  # Detrend
            calcium_dfof = self.normalize(calcium_detrend)  # Normalize
            calcium_smooth = self.smooth_calcium_trace(calcium_dfof, windowlen=self.smooth_window)  # Smooth data

            self.plot_mean_std_historgram(calcium_dfof, title_figure='DFF Data', pdf=pp)
            self.plot_calcium_signals(calcium_smooth, pdf=pp)  # Plot data

            print 'Fish Name ...%s, NUmber of cells.. %s' % (fish[0], np.shape(calcium_smooth))
            calcium_data_for_saving = np.vstack((fish, x, y, row, calcium_smooth))
            calcium_data_allfish = np.hstack((calcium_data_allfish,
                                              calcium_data_for_saving)) if calcium_data_allfish.size else calcium_data_for_saving

            np.save(self.Result_Directory + filesep + ii[:-4], calcium_data_for_saving)
            # print 'Before Shape %s After Shape %s' % (np.shape(alldata['CellZ5'][11:, :]), np.shape(calcium))
        np.save(self.Result_Directory + filesep + 'All', calcium_data_allfish)
        pp.close()

    def normalize(self, calicum_trace):
        # mean = np.mean(calicum_trace[self.baseline[0]:self.baseline[1], :], 0)
        # standarddeviation = np.std(calicum_trace[self.baseline[0]:self.baseline[1], :], 0)
        mean = np.mean(calicum_trace, 0)
        standarddeviation = np.std(calicum_trace, 0)
        normalizedcalcium = (calicum_trace - mean) / (standarddeviation + 0.001)

        return normalizedcalcium

    def remove_neuropils(self, data):
        # Seperate neuropil data
        x = data['CellZ5'][1, :]
        y = data['CellZ5'][2, :]
        calcium = data['CellZ5'][11:, :]
        row = data['CellZ5'][3, :]
        neuropil = data['NeuropilList']
        habenula_or_not = data['AllCoordinates']

        calcium_from_cells = np.array([])
        row_cells = np.array([])
        x_cells = np.array([])
        y_cells = np.array([])

        for jj in xrange(0, 5):
            condition = np.intersect1d(np.where(neuropil[0][jj] == 0)[0],
                                       np.where(habenula_or_not[0, jj][6, :] == 1)[0])
            C1 = calcium[:, row == jj + 1]
            C2 = C1[:, condition]

            R1 = row[row == jj + 1]
            R2 = R1[condition]

            X1 = x[row == jj + 1]
            X2 = X1[condition]

            Y1 = y[row == jj + 1]
            Y2 = Y1[condition]

            calcium_from_cells = np.hstack((calcium_from_cells, C2)) if calcium_from_cells.size else C2
            row_cells = np.hstack((row_cells, R2)) if row_cells.size else R2
            x_cells = np.hstack((x_cells, X2)) if x_cells.size else X2
            y_cells = np.hstack((y_cells, Y2)) if y_cells.size else Y2

        return row_cells, x_cells, y_cells, calcium_from_cells

    def detrend_calcium_trace(self, calcium_trace):
        detrended = np.zeros(np.shape(calcium_trace))
        for ss in xrange(0, np.size(calcium_trace, 1)):
            detrended[:, ss] = detrend(calcium_trace[:, ss])
        return detrended

    def smooth_calcium_trace(self, calcium_trace, windowlen):
        smoothed_calcium_trace = np.zeros(np.shape(calcium_trace))
        for ss in xrange(0, np.size(calcium_trace, 1)):
            smoothed_calcium_trace[:, ss] = PlottingTools.smooth_hanning(calcium_trace[:, ss], windowlen, window='flat')
        return smoothed_calcium_trace

    def plot_mean_std_historgram(self, calcium_trace, title_figure, pdf):

        standard_deviation = np.std(calcium_trace, axis=0)
        mean = np.mean(calcium_trace, axis=0)

        fs, axes = plt.subplots(2, 1)
        axes[0].plot(mean, standard_deviation, 'o', alpha=0.5, markersize=5, markeredgewidth=1)
        PlottingTools.format_axis(axis_handle=axes[0], title=title_figure, xlabel='Mean', ylabel='Standard Deviation')

        axes[1].hist(standard_deviation)
        PlottingTools.format_axis(axis_handle=axes[1], xlabel='Distribution of variance')

        pdf.savefig(fs, bbox_inches='tight')
        plt.close()

    def plot_calcium_signals(self, calcium_trace, pdf):

        # Plot some stuff
        with sns.axes_style('dark'):
            fs = plt.figure()
            gs = plt.GridSpec(2, 1, height_ratios=[2, 1])

            ax1 = fs.add_subplot(gs[0, :])
            plt.imshow(calcium_trace.T, cmap='seismic', aspect='auto', vmin=-2, vmax=2)
            plt.colorbar()
            PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
            PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
            PlottingTools.format_axis(axis_handle=ax1, xlabel='Time (seconds)', ylabel='Cell#',
                                      ylim=[0, np.size(calcium_trace, 1)],
                                      xlim=[0, np.size(calcium_trace, 0)])

            ax2 = fs.add_subplot(gs[1, :])
            plt.plot(np.mean(calcium_trace, 1))
            plt.xlim((0, np.size(calcium_trace, 0)))
            PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
            PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
            PlottingTools.plot_stimulus_patch(self.stimulus_on_time, self.stimulus_off_time, self.stimulus_types,
                                              axis_handle=ax2)
            PlottingTools.format_axis(axis_handle=ax2, xlabel='Time (seconds)', ylabel=r'$\delta$' + 'F/F')
            pdf.savefig(fs, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    # User input
    WorkingDirectory = '/Users/seetha/Desktop/SingleCellAnalysis/Data/Habenula_Matfiles/'
    stimulus_on_time = [46, 95, 139, 194]
    stimulus_off_time = [65, 117, 161, 213]
    stimulus_on_time = [jj - 2 for jj in stimulus_on_time]
    stimulus_off_time = [jj - 2 for jj in stimulus_off_time]
    stimulus_types = ['Blue', 'Blue', 'Blue', 'Blue']

    dff_baseline = [10, 20]
    smooth_window = 5

    responsetype_label = ['ON', 'OFF', 'Inhibitory', 'NoResponse']
    responsetype_idx = range(0, len(responsetype_label))
    responsetype_amplitude = 2

    Clustering_Figures_Directory = os.path.join(WorkingDirectory, 'Clustering', 'ClusteringSignals')
    Clustering_Results_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Cellstats')
    ExperimentParameters_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Experiment_Parameters')

    if not os.path.exists(Clustering_Figures_Directory):
        os.makedirs(Clustering_Figures_Directory)
    if not os.path.exists(Clustering_Results_Directory):
        os.makedirs(Clustering_Results_Directory)
    if not os.path.exists(ExperimentParameters_Directory):
        os.makedirs(ExperimentParameters_Directory)

    Experiment_Parameters = {'stimulus_on_time': stimulus_on_time, 'stimulus_off_time': stimulus_off_time,
                             'stimulus_types': stimulus_types, 'responsetype_label': responsetype_label,
                             'responsetype_idx': responsetype_idx, 'dff_baseline': dff_baseline}
    np.save(ExperimentParameters_Directory + filesep + 'Experiment_Parameter.npy', Experiment_Parameters)

    Get_Compile_Data(WorkingDirectory, Clustering_Results_Directory, Clustering_Figures_Directory, stimulus_on_time,
                     stimulus_off_time, stimulus_types, dff_baseline, smooth_window).LoadData()
