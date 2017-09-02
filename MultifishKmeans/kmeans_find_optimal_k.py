import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import copy
from scipy.cluster.vq import kmeans, vq
from scipy.stats import sem
from scipy.spatial.distance import pdist
from Tools import PlottingTools
from matplotlib.backends.backend_pdf import PdfPages

sns.set_context('paper', font_scale=1.2)
filesep = os.path.sep


class ClusterwithKmeans(object):
    def __init__(self, WorkingDirectory, Result_Directory, Figure_Directory, Exp_Directory, total_clusters,
                 threshold_ampitude, amplitude_trace):

        self.WorkingDirectory = WorkingDirectory
        self.Result_Directory = Result_Directory
        self.Figure_Directory = Figure_Directory
        self.amplitude_trace = amplitude_trace
        self.threshold_ampitude = threshold_ampitude
        self.total_clusters = total_clusters

        temp = np.load(os.path.join(Exp_Directory, 'Experiment_Parameter.npy')).item()
        self.stimulus_on_time = temp['stimulus_on_time']
        self.stimulus_off_time = temp['stimulus_off_time']
        self.responsetypeindex = temp['responsetype_idx']
        self.responsetypelabel = temp['responsetype_label']
        self.stimulus_types = temp['stimulus_types']

    def Kmeans(self):
        AllData = np.load(os.path.join(self.Result_Directory, 'All.npy'))

        fish = AllData[0, :]
        x = AllData[1, :]
        y = AllData[2, :]
        z = AllData[3, :]
        calcium = AllData[4:, :]

        experiment_time = np.size(calcium, 0)
        number_of_cells = np.size(calcium, 1)

        # Threshold data to remove all traces below 0.5 before doing kmeans
        calcium_thresholded, thresholded_cell_idx, calcium_removed, removed_cell_idx = self.threshold_data(
            calcium, number_of_cells)
        number_of_cells_thresholded = np.size(calcium_thresholded, 1)
        print 'Number of removed cells - Before %s, After %s' % (number_of_cells, number_of_cells_thresholded)

        cluster = list()
        sum_dist_cluster = np.array([])
        count = 0
        for numclusters in xrange(1, self.total_clusters, 10):
            print 'Creating .., Cluster Number..', numclusters
            clustercentroids, clusterlabels = self.run_kmeans(calcium_thresholded.T, numclusters)  # Run Kmeans
            self.plot_meankmeanclusters(calcium_thresholded.T, clusterlabels, numclusters,
                                        filename='All')  # Plot clusters

            cluster.append(numclusters)
            temp = self.get_distance_between_mean_and_data(calcium_thresholded.T, clusterlabels, numclusters)
            sum_dist_cluster = np.vstack((sum_dist_cluster, temp)) if sum_dist_cluster.size else temp
            print 'Distance for cluster %s is %s' % (cluster[count], temp)
            count += 1

        sum_dist_cluster = np.vstack((np.asarray(cluster), np.squeeze(sum_dist_cluster)))
        print np.shape(sum_dist_cluster)
        self.plot_sum_distance(sum_dist_cluster, filename='All')

        # np.savetxt(self.Result_Directory + filesep + 'All Fish.txt', final_data, fmt='%0.4f', delimiter='\t')

    def plot_sum_distance(self, sum_distance, filename):
        fs = plt.figure(figsize=(5, 5))
        ax1 = fs.add_subplot(211)
        plt.plot(sum_distance[0, :], sum_distance[1, :], 's-', markersize=10, linewidth=2)
        PlottingTools.format_axis(axis_handle=ax1, xlabel='Clusters', ylabel='Sum Square Distance',
                                  xlim=(0, sum_distance[0, -1]))
        ax1 = fs.add_subplot(212)
        plt.plot(sum_distance[0, 1:], sum_distance[1, 1:], 's-', markersize=10, linewidth=2)
        PlottingTools.format_axis(axis_handle=ax1, xlabel='Clusters', ylabel='Sum Square Distance',
                                  xlim=(0, sum_distance[0, -1]))
        plt.tight_layout()
        plt.savefig(os.path.join(self.Figure_Directory, filename + '_sumsquaredistance' + '.pdf'), bbox_inches='tight')

    def run_kmeans(self, data, numcluster):
        centroids, _ = kmeans(data, numcluster)  # computing K-Means
        idx, _ = vq(data, centroids)  # assign each sample to a cluster
        return centroids, idx

    def plot_meankmeanclusters(self, data, idx, numcluster, filename):
        # some plotting using numpy's logical indexing
        pp = PdfPages(os.path.join(self.Figure_Directory, filename + '_Cluster' + str(numcluster) + '.pdf'))
        num_such_clusters = np.zeros((numcluster, 1))
        colors = sns.color_palette("Set1", numcluster)
        count = 0

        if numcluster > 4:
            num_plots_per_subplot = 4
        else:
            num_plots_per_subplot = numcluster

        for jj in xrange(0, numcluster + 1):

            if (np.mod(jj, num_plots_per_subplot) == 0 and jj != 0) or jj == numcluster:
                PlottingTools.format_legend(axis_handle=ax1)

                PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
                PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
                PlottingTools.plot_stimulus_patch(self.stimulus_on_time, self.stimulus_off_time,
                                                  self.stimulus_types, axis_handle=ax1)

            if jj == 0:
                fs = plt.figure()
                count = 1
            elif jj == numcluster:
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
            if np.size(data[idx == jj, :]) == 0:
                x = np.linspace(0, np.size(data, 1), np.size(data, 1))
                y = np.zeros(np.size(data, 1))
                ax1.plot(x, y, label='Cluster %s Cells %s' % (jj, 0),
                         color=colors[np.int(jj)], linewidth=3, alpha=0.5)
                continue

            num_such_clusters[jj] = np.shape(data[idx == jj, :])[0]
            x = np.linspace(0, np.size(data, 1), np.size(data, 1))
            y = np.mean(data[idx == jj, :], 0)
            error = sem(data[idx == jj, :], 0)
            ax1.plot(x, y, label='Cluster %s Cells %s' % (jj, num_such_clusters[jj]),
                     color=colors[np.int(jj)], linewidth=3, alpha=0.5)
            ax1.fill_between(x, y - error, y + error, color=colors[np.int(jj)], alpha=0.5)
            PlottingTools.format_axis(axis_handle=ax1, xlim=[0, np.size(data, 1)],
                                      ylim=[np.min(y[:]) - 5, 10])

    def get_distance_between_mean_and_data(self, data, idx, numcluster):
        dist = np.zeros(numcluster)
        for jj in xrange(0, numcluster):
            y = np.mean(data[idx == jj, :], 0)
            each_data = data[idx == jj, :]
            if np.size(each_data, 0) == 0:
                data[jj] = 0
            else:
                for kk in xrange(0, np.size(each_data, 0)):
                    dist[jj] += pdist([each_data[kk, :], y])

        sumsquaredist = np.sum(dist[:])

        return sumsquaredist

    def threshold_data(self, data, number_of_cells):
        cell_removed = list()
        cell_thresholded = list()
        data_thresholded = np.array([])
        data_removed = np.array([])

        for cell in xrange(0, number_of_cells):
            if np.max(data[:, cell]) < self.threshold_ampitude:
                cell_removed.append(cell)
                data_removed = np.vstack((data_removed, data[:, cell])) if data_removed.size else data[:, cell]

            elif np.max(data[:self.stimulus_on_time[0], cell]) > 2:
                cell_removed.append(cell)
                data_removed = np.vstack((data_removed, data[:, cell])) if data_removed.size else data[:, cell]

            elif np.max(data[self.stimulus_on_time[0]:self.stimulus_off_time[-1], cell]) < self.threshold_ampitude:
                cell_removed.append(cell)
                data_removed = np.vstack((data_removed, data[:, cell])) if data_removed.size else data[:, cell]

            else:
                cell_thresholded.append(cell)
                data_thresholded = np.vstack((data_thresholded, data[:, cell])) if data_thresholded.size else data[:,
                                                                                                              cell]
        return data_thresholded.T, cell_thresholded, data_removed.T, cell_removed


if __name__ == '__main__':
    # User input
    WorkingDirectory = '/Users/seetha/Desktop/SingleCellAnalysis/Data/Habenula_Matfiles/'

    threshold_ampitude = 0.5
    responsetype_amplitude = 2
    total_clusters = 101

    Clustering_Figures_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Clustering_SEM')
    Clustering_Results_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Cellstats')
    ExperimentParameters_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Experiment_Parameters')

    if not os.path.exists(Clustering_Figures_Directory):
        os.makedirs(Clustering_Figures_Directory)
    if not os.path.exists(Clustering_Results_Directory):
        os.makedirs(Clustering_Results_Directory)
    if not os.path.exists(ExperimentParameters_Directory):
        os.makedirs(ExperimentParameters_Directory)

    ClusterwithKmeans(WorkingDirectory, Clustering_Results_Directory, Clustering_Figures_Directory,
                      ExperimentParameters_Directory, total_clusters, threshold_ampitude,
                      responsetype_amplitude).Kmeans()
