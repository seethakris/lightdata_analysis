""" Plot obtained K-Means """

__author__ = "Seetha Krishnan"
__copyright__ = "Copyright (C) 2016 Seetha Krishnan"
__license__ = "Public Domain"
__version__ = "1.0"


from numpy import load, size, min, max, array, shape, mean, linspace, around, arange, tile, asarray, transpose, zeros, \
    round, reshape, where, r_, convolve, ones
import seaborn as sns
import matplotlib.pyplot as plt
from thunder import Colorize
import time
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl
from copy import copy
from scipy.stats.stats import pearsonr


class plotKmeans(object):
    def __init__(self, fileName, frames_per_sec, stimulus_on_time, stimulus_off_time, stimulus_train, time_start=0,
                 time_end=300):

        self.frames_per_sec = frames_per_sec

        npzfile = load(fileName + 'kmeans_results.npz')
        print 'Files loaded are %s' % npzfile.files
        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.stimulus_train = stimulus_train
        self.kmeans_clusters = npzfile['kmeans_clusters']
        self.kmeans_clusters = self.kmeans_clusters[time_start:time_end, :]
        self.removeclusters = npzfile['ignore_clusters']
        self.brainmap = npzfile['brainmap']
        self.centered_cmap = npzfile['centered_cmap']
        self.img_sim = npzfile['img_sim']
        self.img_labels = npzfile['img_labels']
        self.reference_image = npzfile['reference_image']
        self.matched_pixels = npzfile['matched_pixels']

        print shape(self.brainmap)

        print self.matched_pixels

    def plot_kmeans_components(self, fig1, gs, plot_seperately=0, model_center=1, smooth_window=0, gridspecs='[0,0]',
                               patch=0, **kwargs):

        with sns.axes_style('dark'):
            ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')

            if model_center == 1:  # Models colors according to trace
                if 'cmap' in kwargs:
                    clrs = kwargs['cmap']
                else:
                    clrs = self.centered_cmap

                newclrs = []
                updated_kmeans_cluster = []

                for ii in xrange(0, size(self.kmeans_clusters, 1)):
                    if ii not in self.removeclusters:
                        newclrs.append(clrs[ii])

                        if smooth_window:
                            updated_kmeans_cluster.append(self.smooth_func(self.kmeans_clusters[:, ii], smooth_window))
                        else:
                            updated_kmeans_cluster.append(self.kmeans_clusters[:, ii])
                updated_kmeans_cluster = asarray(updated_kmeans_cluster)
                updated_kmeans_cluster = updated_kmeans_cluster.T

                clrs_cmap = ListedColormap(clrs)

            else:
                clrs_cmap = ListedColormap(sns.color_palette("Paired", n_colors=size(self.kmeans_clusters, 1)),
                                           name='from_list')
                updated_kmeans_cluster = self.kmeans_clusters
                newclrs = clrs_cmap.colors

            # remove those clusters that are ignored before plotting

            if plot_seperately == 0:
                for ii in xrange(0, size(updated_kmeans_cluster, 1)):
                    plt.plot(updated_kmeans_cluster[:, ii], alpha=0.8, lw=3, label=str(ii), color=newclrs[ii])
            else:
                for ii in plot_seperately:
                    plt.plot(updated_kmeans_cluster[:, ii], alpha=0.5, lw=3, label=str(ii), color=newclrs[ii])

            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.locator_params(axis='y', nbins=6)
            ax1.set(xlabel="Time (seconds)", ylabel="Zscore")
            plt.ylim((min(updated_kmeans_cluster) - 0.5, max(updated_kmeans_cluster)))
            plt.xlim((0, size(updated_kmeans_cluster, 0)))
            plt.axhline(y=0, linestyle='-', color='k', linewidth=1)

            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            self.convert_frames_to_sec(fig1, ax1)

        if patch == 1:
            self.plot_stimulus_patch(ax1)

        if plot_seperately == 0:
            return clrs_cmap.colors, updated_kmeans_cluster

    def find_correlation_between_traces(self, updated_kmeans):
        correlation_mat = zeros((size(updated_kmeans, 1), size(updated_kmeans, 1)))
        pvalue = zeros((size(updated_kmeans, 1), size(updated_kmeans, 1)))

        for ii in xrange(0, size(updated_kmeans, 1)):
            for jj in xrange(0, size(updated_kmeans, 1)):
                correlation_mat[ii, jj], pvalue[ii, jj] = pearsonr(updated_kmeans[:, ii], updated_kmeans[:, jj])
        return correlation_mat, pvalue

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
        ax1.set_ylim((y_lim[0], y_lim[1] + y_tick_width / 3))

        for ii in xrange(0, size(self.stimulus_on_time)):
            # Find time of stimulus for width of patch
            time_stim = self.stimulus_off_time[ii] - self.stimulus_on_time[ii]

            # Check different cases of stimuli to create patches
            if self.stimulus_train[ii] == "Blue":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim,
                                          y_tick_width / 4, fc='b')
                ax1.add_patch(rectangle)
                ax1.add_patch(rectangle)
            elif self.stimulus_train[ii] == "Red":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim,
                                          y_tick_width / 4, fc='red')
                ax1.add_patch(rectangle)

    def plot_stimulus_patch_withtimebin(self, fig1, gs, height=5, bins=1, colors='winter', gridspecs='[0,0]',
                                        stimtype='ON'):

        frames_per_bin = int(around(bins * self.frames_per_sec))
        ax2 = eval('fig1.add_subplot(gs' + gridspecs + ')')

        for ii in xrange(0, size(self.stimulus_on_time)):
            if stimtype == "ON":
                total_frames_stimulus = range(self.stimulus_on_time[ii], self.stimulus_off_time[ii])
            else:
                if ii != size(self.stimulus_on_time) - 1:
                    total_frames_stimulus = range(self.stimulus_off_time[ii], self.stimulus_on_time[ii + 1])

            number_of_bins_in_stimulus = size(total_frames_stimulus, 0) / frames_per_bin
            color_map_mat = sns.color_palette(colors, number_of_bins_in_stimulus + 2)[2:]
            count = number_of_bins_in_stimulus - 1

            for jj in xrange(0, size(total_frames_stimulus, 0), frames_per_bin):
                ax2.plot(tile(total_frames_stimulus[jj], height), linspace(0, height, height), '.',
                         color=color_map_mat[count],
                         markersize=10)
                count -= 1
        plt.xlim(0, size(self.kmeans_clusters, 0))
        plt.axis('off')

    def createbrainmap_withcmap(self, fig1, gs, mixing_parameter, gridspecs, **kwargs):
        if 'cmap' in kwargs:
            colormap = ListedColormap(list(kwargs['cmap']), name='braincmap')
        else:
            colormap = ListedColormap(list(self.centered_cmap), name='braincmap')

        brainmap = Colorize(cmap=colormap).transform(self.img_labels, mask=self.img_sim,
                                                     background=self.reference_image,
                                                     mixing=mixing_parameter)
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        ax1.imshow(brainmap)
        ax1.axis('off')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        return brainmap

    def createbrainmap_withcmap_forcombining(self, fig1, gs, gridspecs='[0, :]', **kwargs):
        if 'cmap' in kwargs:
            colormap = ListedColormap(list(kwargs['cmap']), name='braincmap')
        else:
            colormap = ListedColormap(list(self.centered_cmap), name='braincmap')

        brainmap = Colorize(cmap=colormap).transform(self.img_labels, mask=self.img_sim)

        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        ax1.imshow(brainmap)
        ax1.axis('off')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        return brainmap

    def convert_frames_to_sec(self, fig1, ax1):

        n = size(self.kmeans_clusters, 0)
        t = 1.0 / self.frames_per_sec
        xlabels_time = linspace(0, n * t, n + 1)
        fig1.canvas.draw()
        labels = [item.get_text() for item in ax1.get_xticklabels()]
        print labels
        labels_new = [str(around(xlabels_time[int(item)])).rstrip('0').rstrip('.') for item in labels if
                      item != '']
        ax1.set_xticklabels(labels_new)

    def find_matched_pixels_for_new_cmap(self, newclrs, newmap, num_planes, size_x, size_y):

        # Get specific color matches across animals and get mean and standard deviation
        round_clrs = round(newclrs)
        new_array = [tuple(row) for row in round_clrs]
        unique_clrs = (list(set(new_array)))  # Get unique combination of colors

        ## remove black color if it exists
        elem = (0, 0, 0)
        unique_clrs = [value for key, value in enumerate(unique_clrs) if elem != value]
        unique_clrs = round(unique_clrs)

        offset = 5
        newmap_reshaped = zeros((size(newmap, 0) + 2 * offset, size(newmap, 1) + 2 * offset, 3))
        newmap_reshaped[offset:-offset, offset:-offset, :] = newmap
        print shape(newmap_reshaped)

        brainmap_planes = zeros((size_x, size_y, num_planes, 3))

        count_x = 0
        count_y = 0

        for ii in xrange(0, num_planes):
            print count_x, count_y

            if count_y >= size(newmap_reshaped, 1):
                print 'Next Row'
                count_y = 0
                count_x += size_x

            brainmap_planes[:, :, ii, :] = newmap_reshaped[count_x:count_x + size_x, count_y:count_y + size_y, :]
            count_y += size_y

        array_maps = brainmap_planes
        matched_pixels = zeros((size(unique_clrs, 0), num_planes))

        for ii in xrange(0, num_planes):
            array_maps_plane = reshape(array_maps[:, :, ii, :], (size(array_maps, 0) * size(array_maps, 1), 3))
            matched_pixels[:, ii] = [size(where((array(round(array_maps_plane)) == match).all(axis=1))) for match in
                                     unique_clrs]

            print 'Number of pixels in this plane, %s, color is %s, is %s' % (ii, unique_clrs, matched_pixels[:, ii])

        return matched_pixels

    @staticmethod
    def smooth_func(x, window_len=10):
        s = r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        #         w = np.hanning(window_len)
        w = ones(window_len, 'd')
        y = convolve(w / w.sum(), s, mode='valid')
        # print 'Size of y...', shape(y)
        return y[window_len / 2:-window_len / 2 + 1]
