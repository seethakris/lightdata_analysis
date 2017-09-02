import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from Tools import PlottingTools

filesep = os.path.sep
sns.set_context('paper', font_scale=1.2)


class FixclustersandCorrelate(object):
    def __init__(self, WorkingDirectory, ResultsDirectory, FiguresDirectory, ExperimentParameters, switchclusters):
        self.WorkingDirectory = WorkingDirectory
        self.ResultsDirectory = ResultsDirectory
        self.FiguresDirectory = FiguresDirectory
        self.stimulus_on_time = ExperimentParameters['stimulus_on_time']
        self.stimulus_off_time = ExperimentParameters['stimulus_off_time']
        self.responsetypeindex = ExperimentParameters['responsetype_idx']
        self.responsetypelabel = ExperimentParameters['responsetype_label']
        self.stimulus_types = ExperimentParameters['stimulus_types']
        self.numclusters = ExperimentParameters['numclusters']
        self.switchclusters = switchclusters

    def main_function(self):
        # Load results
        dataframe = pd.read_csv(os.path.join(self.ResultsDirectory, 'All Fish.txt'), header=None, sep='\t').values

        fish_name = dataframe[0, :]
        x_coord = dataframe[1, :]
        y_coord = dataframe[2, :]
        cluster_classified_id = dataframe[5, :]
        cluster_original_id = dataframe[6, :]
        kmeans_data = dataframe[7:, :]

        print 'Data Size..', np.shape(kmeans_data)

        cluster_classified_id = self.switch_and_fix_clusters(cluster_original_id, cluster_classified_id)
        # Save updated file
        dataframe[5, :] = cluster_classified_id
        np.savetxt(self.ResultsDirectory + filesep + 'All_Fish_updated.txt', dataframe, fmt='%0.4f', delimiter='\t')

        print 'Plotting by Classified cluster '
        # self.plot_by_clusters(kmeans_data, cluster_classified_id, 'All Fish')
        print 'Plotting by kmeans cluster '
        # self.plot_by_clusters_byoriginal(kmeans_data, cluster_original_id, cluster_classified_id, 'All Fish')
        print 'Plot Correlation Coefficient'
        correlationcoeff, data_noresponses, cluster_classified_noresponses = self.plot_correlation_coefficient(
            kmeans_data, fish_name, cluster_classified_id, 'All Fish')
        classified_responses_updated = self.get_high_correlation_between_on_and_off(correlationcoeff, data_noresponses,
                                                                                    cluster_classified_noresponses,
                                                                                    'All Fish')

        self.plot_correlation_coefficient(data_noresponses, fish_name, classified_responses_updated, 'Temp')

        print 'Plot Updated Cluster Traces'
        self.plot_clustertraces(kmeans_data.T, cluster_original_id, cluster_classified_id, 'All Fish')

    def plot_clustertraces(self, data, idx, classifiedidx, filename):

        responsetype_classification = classifiedidx
        kmeans_classification = idx
        num_cluster_noresponse = np.size(np.unique(
            kmeans_classification[responsetype_classification == self.responsetypeindex[-1]]))
        pp = PdfPages(os.path.join(self.FiguresDirectory, filename + '_kmeansbyclassification.pdf'))
        colors = sns.color_palette("Paired", self.numclusters - num_cluster_noresponse + 5)

        count = 0
        with sns.axes_style('dark'):
            for ii in xrange(0, np.size(self.responsetypeindex)):

                if ii == 0:
                    fs = plt.figure(figsize=(5, 3))
                    count_subplot = 1
                elif ii == np.size(self.responsetypeindex) - 1:
                    pp.savefig(fs, bbox_inches='tight')
                    plt.close()
                    pp.close()
                    break
                elif count_subplot == 2:
                    pp.savefig(fs, bbox_inches='tight')
                    plt.close()
                    fs = plt.figure(figsize=(5, 3))
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
                             color=colors[count], linewidth=0.5)
                    ax1.fill_between(x, y - error, y + error, color=colors[count], alpha=0.3)
                    count += 1

                PlottingTools.format_axis(axis_handle=ax1, title=self.responsetypelabel[ii], xlim=[0, np.size(data, 1)],
                                          ylim=[-2, 4], zeroline=1)

                PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
                PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
                PlottingTools.plot_stimulus_patch(self.stimulus_on_time, self.stimulus_off_time, self.stimulus_types,
                                                  axis_handle=ax1)
                count_subplot += 1

    def plot_correlation_coefficient(self, data, fish, classifiedidx, savefilename):

        # Sort data by reponses - remove those without a response
        without_noresponse = np.where(classifiedidx != self.responsetypeindex[-1])[0]
        classifiedidx_noresponse = classifiedidx[without_noresponse]
        data_noresponse = data[self.stimulus_on_time[0]:self.stimulus_off_time[-1], without_noresponse]
        data_noresponse2 = data[:, without_noresponse]

        sorted_idx = np.argsort(classifiedidx_noresponse)
        print np.shape(sorted_idx), np.shape(without_noresponse)
        sorted_data = data_noresponse[:, sorted_idx]
        correlation = np.corrcoef(sorted_data.T)

        classifiedidx_sorted = classifiedidx_noresponse[sorted_idx]

        boundary_points = np.where((classifiedidx_sorted[1:] - classifiedidx_sorted[:-1]) != 0)[0]
        with sns.axes_style('dark'):
            ax1 = plt.subplot(111)
            plt.imshow(correlation, vmin=-1, vmax=1, cmap='seismic', interpolation='gaussian')
            plt.colorbar(ticks=[-1, 0, 1])
            PlottingTools.format_axis(axis_handle=ax1, xlabel='Cell Number', ylabel='Cell Number',
                                      title='Correlation Coefficients')
            for ii in boundary_points:
                plt.axvline(x=ii, linestyle='-', color='k', linewidth=1)
                plt.axhline(y=ii, linestyle='-', color='k', linewidth=1)

            plt.savefig(self.FiguresDirectory + filesep + savefilename + 'corrcoef.pdf', bbox_inches='tight')
            plt.close()

            fs = plt.figure(figsize=(5, 3))
            ax1 = fs.add_subplot(111)
            plt.imshow(data_noresponse.T, vmin=-1, vmax=1, cmap='jet', interpolation='gaussian', aspect='auto',
                       origin='lower')
            stimulus_on_time = [jj - self.stimulus_on_time[0] for jj in self.stimulus_on_time]
            stimulus_off_time = [jj - self.stimulus_on_time[0] for jj in self.stimulus_off_time]
            PlottingTools.plot_vertical_lines_onset(stimulus_on_time)
            PlottingTools.plot_vertical_lines_offset(stimulus_off_time)
            for ii in boundary_points:
                plt.axhline(y=ii, linestyle='-', color='k', linewidth=2)
            PlottingTools.format_axis(axis_handle=ax1, xlim=(0, np.size(data_noresponse, 0)),
                                      ylim=(0, np.size(data_noresponse, 1)),
                                      xlabel='Time(seconds)', ylabel='Cells', title='All Responses')
            plt.colorbar(ticks=[-1, 0, 1])
            plt.savefig(self.FiguresDirectory + filesep + savefilename + 'heatmap.pdf', bbox_inches='tight')
            plt.close()

        return correlation, data_noresponse2, classifiedidx_noresponse

    def get_high_correlation_between_on_and_off(self, correlation, data, classifiedidx, savefilename):
        ON = np.where(classifiedidx == self.responsetypeindex[self.responsetypelabel.index('ON')])[0]
        OFF = np.where(classifiedidx == self.responsetypeindex[self.responsetypelabel.index('OFF')])[0]
        Inh = np.where(classifiedidx == self.responsetypeindex[self.responsetypelabel.index('Inhibitory')])[0]
        print 'Data Size..', np.shape(data)
        print 'NUmber of ON...', np.shape(ON)
        print 'Number of OFF...', np.shape(OFF)
        print 'Number of Inh...', np.shape(Inh)

        stimulus_on_time = self.stimulus_on_time
        stimulus_off_time = self.stimulus_off_time

        pp = PdfPages(os.path.join(self.FiguresDirectory, savefilename + '_onoffcorrelation.pdf'))
        count_negative = np.zeros(np.size(classifiedidx))
        threshold = 300

        for jj in ON:
            for kk in OFF:
                # print jj, kk, correlation[jj, kk]
                if correlation[jj, kk] > 0.2:
                    count_negative[jj] += 1

        for kk in OFF:
            for jj in ON:
                # print jj, kk, correlation[jj, kk]
                if correlation[jj, kk] > 0.2:
                    count_negative[kk] += 1

        count = 0
        color = sns.color_palette("Paired", 6)
        fs = plt.figure(figsize=(4, 2))
        ax1 = fs.add_subplot(111)
        print 'Size of positive correlations .. %s \n, Total Number of cells %s \n' % (
        np.shape(np.where(count_negative > threshold)[0]), (np.size(ON) + np.size(OFF)))
        for ii in np.where(count_negative > threshold)[0]:
            # print 'Plotting..', ii, count

            plt.plot(data[:, ii], linewidth=2, alpha=0.5, color=color[count], label='Cell %s' % ii)
            count += 1

            if count == 5:
                PlottingTools.format_legend(axis_handle=ax1)
                PlottingTools.plot_vertical_lines_onset(stimulus_on_time)
                PlottingTools.plot_vertical_lines_offset(stimulus_off_time)
                pp.savefig(bbox_inches='tight')
                plt.close()
                count = 0
                fs = plt.figure(figsize=(4, 2))
                ax1 = fs.add_subplot(111)

        data_classified = data[:, np.where(count_negative > threshold)[0]]
        plt.imshow(data_classified.T, vmin=-1, vmax=1, cmap='jet', interpolation='gaussian', aspect='auto')
        PlottingTools.plot_vertical_lines_onset(stimulus_on_time)
        PlottingTools.plot_vertical_lines_offset(stimulus_off_time)
        PlottingTools.format_axis(axis_handle=ax1, xlim=(0, np.size(data, 0)), ylim=(0, np.size(data_classified, 1)),
                                  xlabel='Time(seconds)', ylabel='Cells',
                                  title='Responses where ON and OFF are correlated')
        plt.grid('off')
        plt.colorbar(ticks=[-1, 0, 1])
        pp.savefig(bbox_inches='tight')
        plt.close()
        classifiedidx[np.where(count_negative > threshold)[0]] = self.responsetypeindex[
            self.responsetypelabel.index('NoResponse')]

        plt.close()
        pp.close()

        return classifiedidx

    def switch_and_fix_clusters(self, idx, classifiedidx):
        for key, value in self.switchclusters.iteritems():
            for ii in value:
                print 'Num cells in changed cluster ..', np.size(np.where(classifiedidx[idx == ii])[0])
                classifiedidx[idx == ii] = self.responsetypeindex[self.responsetypelabel.index(key)]

        return classifiedidx

    def plot_by_clusters(self, data, idx, savefilename):

        pp = PdfPages(self.FiguresDirectory + filesep + savefilename + '.pdf')

        for ii in self.responsetypeindex:
            cluster_indices = np.where(idx == ii)[0]

            if np.size(cluster_indices):
                with sns.axes_style('darkgrid'):
                    fs = plt.figure(figsize=(5, 3))
                    ax1 = plt.subplot(111)
                    plt.plot(data[:, cluster_indices], alpha=0.5)
                    plt.plot(np.mean(data[:, cluster_indices], 1), 'k', linewidth=2)
                    plt.title(self.responsetypelabel[ii])
                    PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
                    PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
                    PlottingTools.plot_stimulus_patch(self.stimulus_on_time, self.stimulus_off_time,
                                                      self.stimulus_types, axis_handle=ax1)
                    pp.savefig(fs, bbox_inches='tight')

        pp.close()
        plt.close()

    def plot_by_clusters_byoriginal(self, data, idx, classifiedidx, savefilename):
        pp = PdfPages(self.FiguresDirectory + filesep + savefilename + '_perclassified.pdf')

        for ii in iter(np.unique(idx)):
            cluster_indices = np.where(idx == ii)[0]
            this_classifiedidx = int(classifiedidx[cluster_indices[0]])
            # print ii, classifiedidx[cluster_indices]

            if np.size(cluster_indices):
                with sns.axes_style('darkgrid'):
                    fs = plt.figure(figsize=(4, 2))
                    ax1 = plt.subplot(111)

                    plt.plot(data[:, cluster_indices], alpha=0.5, linewidth=2)
                    plt.plot(np.mean(data[:, cluster_indices], 1), 'k', linewidth=2)

                    plt.title('This cluster %s was classified as %s with number of cells %s' % (
                        ii, self.responsetypelabel[this_classifiedidx], np.size(cluster_indices)), fontsize=8)
                    PlottingTools.plot_vertical_lines_onset(self.stimulus_on_time)
                    PlottingTools.plot_vertical_lines_offset(self.stimulus_off_time)
                    PlottingTools.plot_stimulus_patch(self.stimulus_on_time, self.stimulus_off_time,
                                                      self.stimulus_types, axis_handle=ax1)
                    PlottingTools.format_axis(axis_handle=ax1, xlim=[0, np.size(data, 0)])
                    pp.savefig(fs, bbox_inches='tight')
                    plt.close()

        pp.close()


if __name__ == '__main__':
    # User input
    WorkingDirectory = '/Users/seetha/Desktop/SingleCellAnalysis/Data/Habenula_Matfiles/'

    switch_clusters = {'NoResponse': [14, 19, 20, 22, 28, 34, 51, 53, 56, 57, 58],
                       'ON': [],
                       'OFF': [12, 49]}

    Clustering_Figures_Directory = os.path.join(WorkingDirectory, 'Clustering', 'ClusterTraces')
    Clustering_Results_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Cellstats')
    ExperimentParameters_Directory = os.path.join(WorkingDirectory, 'Clustering', 'Experiment_Parameters')

    if not os.path.exists(Clustering_Figures_Directory):
        os.makedirs(Clustering_Figures_Directory)
    if not os.path.exists(Clustering_Results_Directory):
        os.makedirs(Clustering_Results_Directory)

    Experiment_Parameters = np.load(ExperimentParameters_Directory + filesep + 'Experiment_Parameter.npy').item()

    FixclustersandCorrelate(WorkingDirectory, Clustering_Results_Directory, Clustering_Figures_Directory,
                            Experiment_Parameters, switch_clusters).main_function()

    Experiment_Parameters.update({'switch_clusters': switch_clusters})
    print Experiment_Parameters
    np.save(os.path.join(ExperimentParameters_Directory, 'Experiment_Parameter.npy'), Experiment_Parameters)
