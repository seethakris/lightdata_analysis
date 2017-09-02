import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.vq import kmeans, vq
from scipy.stats import sem
from Tools import PlottingTools
from matplotlib.backends.backend_pdf import PdfPages

sns.set_context('paper', font_scale=1.2)
filesep = os.path.sep


class ClusterwithKmeans(object):
    def __init__(self, WorkingDirectory, Result_Directory, Figure_Directory, Exp_Directory, num_clusters,
                 threshold_ampitude,
                 amplitude_trace):
        self.WorkingDirectory = WorkingDirectory
        self.Result_Directory = Result_Directory
        self.Figure_Directory = Figure_Directory
        self.numclusters = num_clusters
        self.amplitude_trace = amplitude_trace
        self.threshold_ampitude = threshold_ampitude

        temp = np.load(os.path.join(Exp_Directory, 'Experiment_Parameter.npy')).item()
        self.stimulus_on_time = temp['stimulus_on_time']
        self.stimulus_off_time = temp['stimulus_off_time']
        self.responsetypeindex = temp['responsetype_idx']
        self.responsetypelabel = temp['responsetype_label']
        self.stimulus_types = temp['stimulus_types']

        temp.update({'numclusters': self.numclusters})
        np.save(os.path.join(Exp_Directory, 'Experiment_Parameter.npy'), temp)

    def Kmeans(self):
        AllData = np.load(os.path.join(self.Result_Directory, 'All.npy'))

        experiment_time = np.size(AllData[4:, :], 0)
        number_of_cells = np.size(AllData[4:, :], 1)

        # Threshold data to remove all traces below 0.5 before doing kmeans
        thresholded, thresholded_cell_idx, removed, removed_cell_idx = self.threshold_data(
            AllData, number_of_cells)

        calcium_removed = removed[4:, :]
        calcium_thresholded = thresholded[4:, :]

        number_of_cells_thresholded = np.size(calcium_thresholded, 1)
        print 'Number of removed cells - Before %s, After %s' % (number_of_cells, number_of_cells_thresholded)

        clustercentroids, clusterlabels = self.run_kmeans(calcium_thresholded.T)  # Run Kmeans
        self.get_low_correlation_traces_within_clusters(calcium_thresholded, clustercentroids, clusterlabels,
                                                        filename='All Fish')

        # self.plot_meankmeanclusters(calcium_thresholded.T, clusterlabels,
        #                             filename='All Fish')  # Plot clusters
        #
        # stimulus_trace_for_correlation = self.traces_for_correlation(experiment_time, filename='All Fish')
        #
        # correlation_coefficient = self.correlate_with_clusters(calcium_thresholded.T, clusterlabels,
        #                                                        stimulus_trace_for_correlation,
        #                                                        filename='All Fish')
        #
        # classified_array = self.classify_cells(thresholded_cell_idx, removed_cell_idx, clusterlabels,
        #                                        correlation_coefficient)
        #
        # updated_data_calcium = np.hstack((calcium_thresholded, calcium_removed))
        # updated_data_rest = np.hstack((thresholded[:4, :], removed[:4, :]))
        # self.plot_clustertraces(updated_data_calcium.T, classified_array, filename='All.mat')
        #
        # print np.shape(updated_data_calcium), np.shape(updated_data_rest), np.shape(classified_array)
        # print removed[:4, :]
        # final_data = np.vstack((updated_data_rest, classified_array, updated_data_calcium))
        # #
        # np.savetxt(self.Result_Directory + filesep + 'All Fish.txt', final_data, fmt='%0.4f', delimiter='\t')

    def run_kmeans(self, data):
        centroids, _ = kmeans(data, self.numclusters)  # computing K-Means
        idx, _ = vq(data, centroids)  # assign each sample to a cluster
        return centroids, idx

    def get_low_correlation_traces_within_clusters(self, data, centroids, idx, filename):

        pp = PdfPages(os.path.join(self.Figure_Directory, filename + '_correlationwithcentroids.pdf'))

        for ii in xrange(0, self.numclusters):
            data_thisidx = data[:, idx == ii]
            fs = plt.figure(figsize=(4, 3))
            ax1 = fs.add_subplot(111)
            PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
            PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
            PlottingTools.format_legend(legend_string=self.responsetypelabel, axis_handle=ax1)
            plt.plot(centroids[ii, :].T, linewidth=2, color='k', label='Cluster Centroid')
            plt.title('Centroid Number %s' % ii)

            for jj in xrange(0, np.size(data_thisidx, 1)):
                correlation = np.corrcoef(data_thisidx[:, jj], centroids[ii, :])[0, 1]
                print ii, jj, correlation
                if correlation < 0:
                    plt.plot(data_thisidx[:, jj], linewidth=1, alpha=0.5,
                             label='Cell %s  Correlation %s' % (jj, correlation))

            PlottingTools.format_legend(axis_handle=ax1)
            pp.savefig(bbox_inches='tight')
            plt.close()

        pp.close()

    def threshold_data(self, alldata, number_of_cells):
        cell_removed = list()
        cell_thresholded = list()
        data_thresholded = np.array([])
        data_removed = np.array([])

        data = alldata[4:, :]

        for cell in xrange(0, number_of_cells):
            if np.max(data[:, cell]) < self.threshold_ampitude:
                cell_removed.append(cell)
                data_removed = np.vstack((data_removed, alldata[:, cell])) if data_removed.size else alldata[:, cell]

            elif np.max(data[self.stimulus_on_time[0]:self.stimulus_off_time[-1], cell]) < self.threshold_ampitude:
                cell_removed.append(cell)
                data_removed = np.vstack((data_removed, alldata[:, cell])) if data_removed.size else alldata[:, cell]

            else:
                cell_thresholded.append(cell)
                data_thresholded = np.vstack(
                    (data_thresholded, alldata[:, cell])) if data_thresholded.size else alldata[:, cell]

        return data_thresholded.T, cell_thresholded, data_removed.T, cell_removed

    def correlate_with_clusters(self, data, idx, stimulustrace, filename):
        correlation_coeff = np.zeros((self.numclusters, np.size(stimulustrace, 1)))
        for ii in xrange(0, self.numclusters):
            datatrace = np.mean(data[idx == ii, :], 0)
            for jj in xrange(0, np.size(stimulustrace, 1)):
                stimulus = stimulustrace[:, jj]
                correlation_coeff[ii, jj] = np.corrcoef(datatrace, stimulus)[0, 1]

        plt.figure()
        plt.imshow(correlation_coeff, cmap='seismic', aspect='auto', interpolation='None', vmin=-1, vmax=1)
        plt.xticks(range(0, len(self.responsetypelabel)), self.responsetypelabel, rotation='vertical')
        plt.grid('off')
        plt.colorbar()
        plt.savefig(os.path.join(self.Figure_Directory, filename + '_correlationcoeff.png'), bbox_inches='tight')
        plt.close()
        return correlation_coeff

    def plot_clustertraces(self, data, classified, filename):

        responsetype_classification = classified[1, :]
        kmeans_classification = classified[2, :]
        pp = PdfPages(os.path.join(self.Figure_Directory, filename + '_kmeansbyclassification.pdf'))
        colors = sns.color_palette("Paired", self.numclusters + 1)

        for ii in xrange(0, np.size(self.responsetypeindex) + 1):

            if ii == 0:
                fs = plt.figure()
                count_subplot = 1
            elif ii == np.size(self.responsetypeindex):
                pp.savefig(fs, bbox_inches='tight')
                plt.close()
                pp.close()
                break
            elif count_subplot == 2:
                pp.savefig(fs, bbox_inches='tight')
                plt.close()
                fs = plt.figure()
                count_subplot = 1

            ax1 = fs.add_subplot(2, 1, count_subplot)
            responsetype = np.where(responsetype_classification == self.responsetypeindex[ii])[0]
            kmeanstype = np.unique(kmeans_classification[responsetype])
            x = np.linspace(0, np.size(data, 1), np.size(data, 1))
            for jj in iter(kmeanstype):
                y = np.mean(data[kmeans_classification == jj, :], 0)
                error = sem(data[kmeans_classification == jj, :], 0)
                num_such_clusters = np.shape(data[kmeans_classification == jj, :])[0]
                ax1.plot(x, y, label='Cluster %s Cells %s' % (jj, num_such_clusters),
                         color=colors[np.int(jj)], linewidth=3)
                ax1.fill_between(x, y - error, y + error, color=colors[np.int(jj)], alpha=0.5)

            PlottingTools.format_legend(axis_handle=ax1)
            PlottingTools.format_axis(axis_handle=ax1, title=self.responsetypelabel[ii], xlim=[0, np.size(data, 1)],
                                      ylim=[np.min(data), np.max(data)])

            PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
            PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
            PlottingTools.plot_stimulus_patch(self.stimulus_on_time, self.stimulus_off_time, self.stimulus_types,
                                              axis_handle=ax1)
            count_subplot += 1

    def traces_for_correlation(self, time, filename):
        stim_traces = np.zeros((time, np.size(self.responsetypelabel)))
        stim_time = self.stimulus_off_time[0] - self.stimulus_on_time[0]

        for ii in xrange(0, np.size(self.responsetypelabel)):
            print self.responsetypelabel[ii]
            for jj in xrange(0, np.size(self.stimulus_on_time)):
                if self.responsetypelabel[ii] == 'ON':
                    stim_traces[self.stimulus_on_time[jj]:self.stimulus_off_time[jj] - stim_time / 2,
                    ii] = self.amplitude_trace
                elif self.responsetypelabel[ii] == 'OFF':
                    if jj == np.size(self.stimulus_on_time) - 1:
                        stim_traces[self.stimulus_off_time[jj]:self.stimulus_off_time[jj] + stim_time,
                        ii] = self.amplitude_trace
                    else:
                        stim_traces[self.stimulus_off_time[jj]:self.stimulus_on_time[jj + 1] - stim_time / 2,
                        ii] = self.amplitude_trace
                elif self.responsetypelabel[ii] == 'Inhibitory':
                    stim_traces[self.stimulus_on_time[jj]:self.stimulus_off_time[jj] - stim_time / 2,
                    ii] = -1.5

        stim_traces_smooth = np.zeros((np.shape(stim_traces)))
        for ii in xrange(0, np.size(self.responsetypelabel)):
            stim_traces_smooth[:, ii] = PlottingTools.smooth_hanning(stim_traces[:, ii], window_len=stim_time / 2,
                                                                     window='hanning')
        ax1 = plt.subplot(111)
        plt.plot(stim_traces_smooth, alpha=0.5, linewidth=2)
        PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
        PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
        PlottingTools.format_legend(legend_string=self.responsetypelabel, axis_handle=ax1)
        PlottingTools.format_axis(axis_handle=ax1, ylim=[-4, self.amplitude_trace + 1])
        plt.savefig(os.path.join(self.Figure_Directory, filename + '_responsetypetraces.png'),
                    bbox_inches='tight')
        plt.close()

        return stim_traces_smooth

    def classify_cells(self, thresholded_cells, removed_cells, idx, correlation_coefficient):  # Data = complete dataset

        print idx
        classified_data = np.zeros((3, len(thresholded_cells)))
        classified_data_removed = np.zeros((3, len(removed_cells)))

        classified_data[0, :] = thresholded_cells
        classified_data[2, :] = idx
        classified_data_removed[0, :] = removed_cells
        classified_data_removed[1, :] = self.responsetypeindex[self.responsetypelabel.index('NoResponse')]
        classified_data_removed[2, :] = self.numclusters

        max_correlation = np.argmax(correlation_coefficient[:, :-1], 1)
        low_correlation_values = (correlation_coefficient[:, :-1] <= 0.2).all(axis=1).astype(int)

        print 'Max correlation', max_correlation
        print 'Low Correlation', low_correlation_values

        for ii in xrange(0, np.size(max_correlation)):
            if low_correlation_values[ii] == 1:
                classified_data[1, idx == ii] = self.responsetypeindex[self.responsetypelabel.index('NoResponse')]
                print ii, 'No Response'
            else:
                # Try out the many scenarios for labeling the traces
                sort_indices = np.argsort(correlation_coefficient[ii, :-1])[::-1]
                get_highest_indices = list(np.where(correlation_coefficient[ii, :-1] > 0.3)[0])
                # get_highest_indices_withlen1 = list(np.where(correlation_coefficient[ii, :-1] >= 0.2)[0])
                print ii, get_highest_indices

                if len(get_highest_indices) == 1:
                    classified_data[1, idx == ii] = get_highest_indices[0]
                    print ii, 'Found ', self.responsetypelabel[get_highest_indices[0]]
                elif get_highest_indices == []:
                    classified_data[1, idx == ii] = self.responsetypeindex[self.responsetypelabel.index('NoResponse')]
                    print ii, 'Found No Response'
                elif get_highest_indices[0] == 2 and correlation_coefficient[
                    ii, self.responsetypeindex[self.responsetypelabel.index('ON')]] < 0:
                    classified_data[1, idx == ii] = self.responsetypeindex[
                        self.responsetypelabel.index('Inhibitory')]
                    print ii, 'Found Inhibitory'
                else:
                    classified_data[1, idx == ii] = get_highest_indices[0]
                    print ii, 'Found Other ', self.responsetypelabel[get_highest_indices[0]]

        final_classified_data = np.hstack((classified_data, classified_data_removed))
        return final_classified_data

    def plot_meankmeanclusters(self, data, idx, filename):
        # some plotting using numpy's logical indexing
        pp = PdfPages(os.path.join(self.Figure_Directory, filename + '_kmeanclusters.pdf'))
        num_such_clusters = np.zeros((self.numclusters, 1))
        colors = sns.color_palette("Set1", self.numclusters)
        count = 0
        num_plots_per_subplot = 5

        for jj in xrange(0, self.numclusters + 1):

            if np.mod(jj, num_plots_per_subplot) == 0 and (jj != 0 or jj == self.numclusters):
                PlottingTools.format_legend(axis_handle=ax1)
                PlottingTools.format_axis(axis_handle=ax1, xlim=[0, np.size(data, 1)],
                                          ylim=[np.min(data), np.max(data)])

                PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
                PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
                PlottingTools.plot_stimulus_patch(self.stimulus_on_time, self.stimulus_off_time,
                                                  self.stimulus_types, axis_handle=ax1)

            if jj == 0:
                fs = plt.figure()
                count = 1
            elif jj == self.numclusters:
                pp.savefig(fs, bbox_inches='tight')
                plt.close()
                pp.close()
                break
            elif np.mod(jj, num_plots_per_subplot) == 0 and count == 1:
                count += 1
            elif np.mod(jj, num_plots_per_subplot) == 0:
                pp.savefig(fs, bbox_inches='tight')
                plt.close()
                fs = plt.figure()
                count = 1

            ax1 = fs.add_subplot(2, 1, count)
            num_such_clusters[jj] = np.shape(data[idx == jj, :])[0]
            ax1.plot(np.mean(data[idx == jj, :], 0),
                     label='Cluster %s Cells %s' % (jj, num_such_clusters[jj]), color=colors[jj], linewidth=3,
                     alpha=0.5)


if __name__ == '__main__':
    # User input
    WorkingDirectory = '/Users/seetha/Desktop/SingleCellAnalysis/Data/Habenula_Matfiles/'

    threshold_ampitude = 2
    responsetype_amplitude = 2
    numclusters = 60

    Clustering_Figures_Directory = os.path.join(WorkingDirectory, 'Clustering', 'ClusteringSignals')
    Clustering_Results_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Cellstats')
    ExperimentParameters_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Experiment_Parameters')

    ClusterwithKmeans(WorkingDirectory, Clustering_Results_Directory, Clustering_Figures_Directory,
                      ExperimentParameters_Directory, numclusters, threshold_ampitude,
                      responsetype_amplitude).Kmeans()
