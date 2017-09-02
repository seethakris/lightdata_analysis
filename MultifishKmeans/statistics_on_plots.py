"""
Stats on kmean clusters
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from collections import Counter
from scipy.stats import sem, norm
from Tools import PlottingTools

filesep = os.path.sep
sns.set_context('paper', font_scale=1.2)


class PlotClusterTraces(object):
    def __init__(self, WorkingDirectory, ResultsDirectory, FiguresDirectory, ExperimentParameters):
        self.WorkingDirectory = WorkingDirectory
        self.ResultsDirectory = ResultsDirectory
        self.FiguresDirectory = FiguresDirectory
        self.stimulus_on_time = ExperimentParameters['stimulus_on_time']
        self.stimulus_off_time = ExperimentParameters['stimulus_off_time']
        self.stimulus_types = ExperimentParameters['stimulus_types']
        self.responsetype_label = ExperimentParameters['responsetype_label']
        self.numclusters = ExperimentParameters['numclusters']
        self.responsetype_index = range(0, len(ExperimentParameters['responsetype_label']))

    def collect_fish_specific_files(self):
        # Load results
        AllData = np.load(os.path.join(self.ResultsDirectory, 'All.npy'))
        dataframe = pd.read_csv(os.path.join(self.ResultsDirectory, 'All Fish.txt'), header=None, sep='\t').values

        pdf_cluster = PdfPages(self.FiguresDirectory + filesep + 'Distance_within_Cluster.pdf')

        fish_name = dataframe[0, :]
        x_coord = dataframe[1, :]
        y_coord = dataframe[2, :]
        z_coord = dataframe[3, :]
        cluster_classified_id = dataframe[5, :]
        cluster_original_id = dataframe[6, :]
        kmeans_data = dataframe[7:, :]

        distance, perc, perc_eachfish = self.get_distance_number_within_clusters(fish_name, x_coord, y_coord, z_coord,
                                                                                 cluster_classified_id)
        print perc_eachfish

        error = sem(perc_eachfish, 0)
        sample_95ci = 1.96 * error

        print 'Mean Number of cells... %s' % np.mean(perc_eachfish, 0)
        print '95 CI is.. %s' % sample_95ci
        print '95 CI Number of cells...lower bound %s' % (np.mean(perc_eachfish, 0) - sample_95ci)
        print '95 CI Number of cells... upper bound %s' % (np.mean(perc_eachfish, 0) + sample_95ci)

        rand_distance = np.asarray(
            self.get_randomised_dist_between_cells(x_coord, y_coord, cluster_classified_id, z_coord))
        self.plot_distance_within_cluster(fish_name, distance, rand_distance, perc, perc_eachfish)

    def plot_distance_within_cluster(self, fish_name, distance, rand_distance, numbercells, numbercells_eachfish):

        pp = PdfPages(os.path.join(self.FiguresDirectory, 'Distance_within_Cluster.pdf'))
        for ii in xrange(0, np.size(np.unique(fish_name)) + 2):
            print ii, np.shape(numbercells)

            fs, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5), sharey='row', sharex='col',
                                                        gridspec_kw={'width_ratios': [3, 1]})

            if ii == np.size(np.unique(fish_name)):
                # Stats per response by fish
                this_distance = np.zeros((np.size(distance, 0) * np.size(distance, 1), np.size(distance, 2)))
                for kk in self.responsetype_index:
                    this_distance[:, kk] = np.ravel(distance[:, :, kk])
                distance_dataframe = pd.DataFrame(this_distance, columns=self.responsetype_label)

                sns.stripplot(data=this_distance, jitter=True, ax=ax3, color=".3")
                sns.boxplot(data=distance_dataframe, ax=ax3, whis=np.inf)

                for nn in xrange(0, np.size(numbercells_eachfish, 0)):
                    ax1.plot(numbercells_eachfish[nn, :], 's-', markersize=6, markeredgecolor='black',
                             label='Fish %s' % np.unique(fish_name)[nn])

                PlottingTools.format_legend(axis_handle=ax1)
                PlottingTools.format_axis(axis_handle=ax1, title='All Fish')
                plt.sca(ax1)
                plt.ylim((0, numbercells_eachfish.max()))

            elif ii == np.size(np.unique(fish_name)) + 1:
                # Stats per fish by response
                this_distance = np.zeros((np.size(distance, 2) * np.size(distance, 1), np.size(distance, 0)))
                for kk in xrange(0, np.size(np.unique(fish_name))):
                    this_distance[:, kk] = np.ravel(distance[kk, :, :])
                distance_dataframe = pd.DataFrame(this_distance, columns=np.unique(fish_name))

                sns.stripplot(data=this_distance, jitter=True, ax=ax3, color=".3")
                sns.boxplot(data=distance_dataframe, ax=ax3, whis=np.inf)

                for jj in xrange(0, np.size(numbercells, 2)):
                    print jj
                    y = np.mean(numbercells[:, :, jj], 1)
                    error = sem(numbercells[:, :, jj], 1)
                    sample_95ci = 1.96 * error

                    x = np.linspace(0, np.size(np.unique(fish_name)), np.size(np.unique(fish_name)), endpoint=False)
                    ax1.errorbar(x, y, yerr=sample_95ci, fmt='-s', markersize=6, markeredgecolor='black',
                                 label=self.responsetype_label[jj])

                    PlottingTools.format_legend(axis_handle=ax1)
                    plt.sca(ax3)
                    plt.xlabel('Fish Number')

            else:
                this_distance = np.squeeze(distance[ii, :, :])
                this_cells = np.squeeze(numbercells[ii, :, :])
                distance_dataframe = pd.DataFrame(this_distance, columns=self.responsetype_label)

                sns.stripplot(data=this_distance, jitter=True, ax=ax3, color=".3")
                sns.factorplot(data=distance_dataframe, ax=ax3, ci=95, estimator=np.median)

                for jj in xrange(0, np.size(this_cells, 0)):
                    ax1.plot(this_cells[jj, :], 's-', markersize=6, markeredgecolor='black',
                             label='Plane ' + str(jj + 1))
                    PlottingTools.format_legend(axis_handle=ax1)
                    PlottingTools.format_axis(axis_handle=ax1, title='Fish %s' % np.unique(fish_name)[ii])

            plt.sca(ax4)
            plt.boxplot(rand_distance[:])
            plt.setp(ax4, xticks=[1], xticklabels=['Random Dist'])

            plt.sca(ax3)
            # plt.ylim((0, np.max(this_distance[:]) + 200))
            plt.xticks(rotation=90)

            plt.sca(ax2)
            plt.axis('off')

            PlottingTools.format_axis(axis_handle=ax3, ylabel='Distance Measure')
            PlottingTools.format_axis(axis_handle=ax1, ylabel='Percentage Cell')

            plt.tight_layout()
            pp.savefig(fs)
        pp.close()

    def get_distance_number_within_clusters(self, fish_name, x_coord, y_coord, z_coord, idx):

        # Get distance and number of cells within cluster by fish  and plane
        distance_cluster = np.empty(
            (np.size(np.unique(fish_name)), np.size(np.unique(z_coord)), np.size(self.responsetype_index)))
        distance_cluster[:] = np.NaN

        perc_cluster = np.zeros(
            (np.size(np.unique(fish_name)), np.size(np.unique(z_coord)), np.size(self.responsetype_index)),
            dtype=np.float)
        num_cells_cluster_fish = np.zeros((np.size(np.unique(fish_name)), np.size(self.responsetype_index)),
                                          dtype=np.float)
        perc_cluster_fish = np.zeros((np.size(np.unique(fish_name)), np.size(self.responsetype_index)),
                                     dtype=np.float)

        count_fish = 0
        for ii in iter(np.unique(fish_name)):

            this_fish = np.where(fish_name == ii)[0]

            for jj in iter(np.unique(z_coord)):

                this_z = np.where(z_coord == jj)[0]
                indices = np.intersect1d(this_z, this_fish)

                for kk in self.responsetype_index:
                    this_cluster = np.where(idx[indices] == kk)

                    perc_cluster[count_fish, int(jj - 1), kk] = np.divide(np.size(this_cluster),
                                                                          np.float(np.size(idx[indices]))) * 100
                    num_cells_cluster_fish[count_fish, kk] += (np.size(this_cluster))

                    print 'Perc fish number %s, response type %s, size_cluster %s, size_thisfish %s' % (
                        count_fish, kk, np.size(this_cluster), np.float(np.size(this_fish)))

                    if np.size(this_cluster):
                        x_cluster = x_coord[this_cluster]
                        y_cluster = y_coord[this_cluster]
                        xy = [x_cluster, y_cluster]
                        distance_cluster[count_fish, int(jj - 1), kk] = pdist(xy, metric='euclidean')

            perc_cluster_fish[count_fish, :] = (num_cells_cluster_fish[count_fish, :] / np.float(
                np.size(this_fish))) * 100
            print 'Perc Per fish is  %s' % (perc_cluster_fish[count_fish, :])

            count_fish += 1
        print np.shape(distance_cluster)
        return distance_cluster, perc_cluster, perc_cluster_fish

    def get_randomised_dist_between_cells(self, x_coord, y_coord, idx, z_coord):
        counter = 0
        distance_cluster = list()
        for jj in iter(np.unique(z_coord)):
            while (counter < 1000):
                x = x_coord[z_coord == jj]
                y = y_coord[z_coord == jj]
                cluster = idx[z_coord == jj]

                xy = np.vstack((x, y)).T
                np.random.shuffle(xy)
                xy = xy.T

                for ii in self.responsetype_index:
                    cluster_indices = np.where(cluster == ii)[0]
                    if cluster_indices.size:
                        xy_random = xy[:, cluster_indices]
                        distance_cluster.append(pdist(xy_random, metric='euclidean'))
                        counter += 1
                    else:
                        continue
        return distance_cluster


if __name__ == '__main__':
    # User input
    WorkingDirectory = WorkingDirectory = '/Users/seetha/Desktop/SingleCellAnalysis/Data/Habenula_Matfiles/'
    ExperimentParameters_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Experiment_Parameters')
    Clustering_Results_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Cellstats')
    Figures_Directory = os.path.join(WorkingDirectory, 'Clustering', 'ActivityStats')

    if not os.path.exists(Figures_Directory):
        os.makedirs(Figures_Directory)

    Experiment_Parameters = np.load(ExperimentParameters_Directory + filesep + 'Experiment_Parameter.npy').item()

    PlotClusterTraces(WorkingDirectory, Clustering_Results_Directory, Figures_Directory,
                      Experiment_Parameters).collect_fish_specific_files()
