""" Run PCA trajectories as colormaps """

__author__ = "Seetha Krishnan"
__copyright__ = "Copyright (C) 2016 Seetha Krishnan"
__license__ = "Public Domain"
__version__ = "1.0"


from numpy import load, size, min, max, array, shape, mean, linspace, around, arange, tile
import seaborn as sns
import matplotlib.pyplot as plt
from thunder import Colorize
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl


class plotPCA(object):
    def __init__(self, fileName, frames_per_sec, color_mat, stimulus_on_time,
                 stimulus_off_time,
                 stimulus_train):

        self.frames_per_sec = frames_per_sec

        # Load file
        npzfile = load(fileName + 'pca_results.npz')
        print 'Files loaded are %s' % npzfile.files
        self.pca_components = npzfile['pca_components']
        self.required_pcs = npzfile['required_pcs']
        self.unique_clrs = npzfile['unique_clrs']
        self.matched_signals = npzfile['mean_signal']
        self.maps = npzfile['maps']

        print 'PCA_components size', shape(self.pca_components)

        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.stimulus_train = stimulus_train
        self.color_mat = color_mat
        self.image = Colorize.image

    def plot_pca_components(self, fig1, gs, plot_title='Habneula', gridspecs='[0,0]'):

        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')

        with sns.axes_style('darkgrid'):
            for ii in range(size(self.pca_components, 1)):
                if ii in self.required_pcs:
                    plt.plot(self.pca_components[:, ii], '-', linewidth=5, label=str(ii))
                else:
                    plt.plot(self.pca_components[:, ii], '--', label=str(ii))

            plt.title(plot_title, fontsize=14)
            sns.axlabel("Time (seconds)", "a.u")
            plt.locator_params(axis='y', nbins=4)
            sns.axlabel("Time (seconds)", "a.u")
            ax1.legend(prop={'size': 14}, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True,
                       shadow=True)
            plt.axhline(y=0, linestyle='-', color='k', linewidth=1)
            ax1.locator_params(axis='y', pad=50, nbins=2)
            plt.ylim((min(self.pca_components) - 0.0001, max(self.pca_components) + 0.0001))
            plt.xlim(0, size(self.pca_components, 0))
            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            self.plot_stimulus_patch(ax1)
            self.convert_frames_to_sec(fig1, ax1)

    def plot_pca_colored_by_timebins(self, fig1, gs, bins, colors='winter', gridspecs1='[0,0]', gridspecs2='[0,0]'):

        frames_per_bin = int(around(bins * self.frames_per_sec))
        if frames_per_bin == 1:
            frames_per_bin = 2

        with sns.axes_style('darkgrid'):
            ax1 = eval('fig1.add_subplot(gs' + gridspecs1 + ')')

            for ii in xrange(0, size(self.stimulus_on_time)):
                total_frames_stimulus = self.pca_components[self.stimulus_on_time[ii]: self.stimulus_off_time[ii] + 1,
                                        :]

                if ii != size(self.stimulus_off_time) - 1:
                    total_frames_stimulus_off = self.pca_components[
                                                self.stimulus_off_time[ii]: self.stimulus_on_time[ii + 1] + 1, :]
                else:
                    total_frames_stimulus_off = self.pca_components[
                                                self.stimulus_off_time[ii]:self.stimulus_off_time[ii] + size(
                                                    total_frames_stimulus, 0) + 2]

                if ii == 0:  # First baseline
                    ax1.plot(self.pca_components[
                             self.stimulus_on_time[ii] - size(total_frames_stimulus, 0):self.stimulus_on_time[ii] + 1,
                             self.required_pcs[0]], self.pca_components[
                                                    self.stimulus_on_time[ii] - size(total_frames_stimulus, 0):
                                                    self.stimulus_on_time[ii] + 1,
                                                    self.required_pcs[1]], color='gray', linewidth=3)

                number_of_bins_in_stimulus_on = (size(total_frames_stimulus, 0) / frames_per_bin)
                color_map_mat_on = sns.color_palette(colors, number_of_bins_in_stimulus_on + 2)[2:]
                count_on = number_of_bins_in_stimulus_on - 1

                print number_of_bins_in_stimulus_on

                number_of_bins_in_stimulus_off = (size(total_frames_stimulus_off, 0) / frames_per_bin)
                color_map_mat_off = sns.color_palette(colors, number_of_bins_in_stimulus_off + 2)[2:]
                count_off = number_of_bins_in_stimulus_off - 1

                print number_of_bins_in_stimulus_off

                for jj in xrange(0, size(total_frames_stimulus, 0), frames_per_bin):
                    ax1.plot(total_frames_stimulus[jj:jj + frames_per_bin + 1, self.required_pcs[0]],
                             total_frames_stimulus[jj:jj + frames_per_bin + 1, self.required_pcs[1]],
                             color=color_map_mat_on[count_on], linewidth=4)
                    count_on -= 1

                for jj in xrange(0, size(total_frames_stimulus_off, 0), frames_per_bin):
                    ax1.plot(total_frames_stimulus_off[jj:jj + frames_per_bin + 1, self.required_pcs[0]],
                             total_frames_stimulus_off[jj:jj + frames_per_bin + 1, self.required_pcs[1]],
                             color=color_map_mat_off[count_off], linestyle='--', linewidth=4)


                    count_off -= 1

                if ii == size(self.stimulus_on_time) - 1:  # Last baseline
                    ax1.plot(self.pca_components[
                             self.stimulus_off_time[ii] + size(total_frames_stimulus, 0) - 1:
                             self.stimulus_off_time[ii] + size(total_frames_stimulus, 0) * 2,
                             self.required_pcs[0]], self.pca_components[
                                                    self.stimulus_off_time[ii] + size(total_frames_stimulus, 0) - 1:
                                                    self.stimulus_off_time[ii] + size(total_frames_stimulus, 0) * 2,
                                                    self.required_pcs[1]], color='black', linewidth=3)
            ax1.set_xlabel('PC' + str(self.required_pcs[0] + 1), linespacing=10)
            ax1.set_ylabel('PC' + str(self.required_pcs[1] + 1), linespacing=10)
            ax1.locator_params(axis='y', nbins=6)
            ax1.locator_params(axis='x', nbins=6)

            if gridspecs2 != 0:
                self.plot_pca_components_bytimebins(fig1, gs, bins, number_of_bins_in_stimulus, color_map_mat,
                                                    frames_per_bin, gridspecs=gridspecs2)

    def plot_pca_components_bytimebins(self, fig1, gs, bins, number_of_bins_in_stimulus, color_map_mat, frames_per_bin,
                                       gridspecs='[0,0]'):

        with sns.axes_style('dark'):
            ax2 = eval('fig1.add_subplot(gs' + gridspecs + ')')
            count_timepnt = self.stimulus_on_time[0]
            ax2.plot(self.pca_components[:, self.required_pcs], 'k', linewidth=0.5)

            for ii in xrange(0, size(self.stimulus_on_time)):

                total_frames_stimulus = self.pca_components[self.stimulus_on_time[ii]: self.stimulus_off_time[ii], :]

                if ii != size(self.stimulus_off_time) - 1:
                    total_frames_stimulus_off = self.pca_components[
                                                self.stimulus_off_time[ii]: self.stimulus_on_time[ii + 1], :]
                else:
                    total_frames_stimulus_off = self.pca_components[
                                                self.stimulus_off_time[ii]:self.stimulus_off_time[ii] + size(
                                                    total_frames_stimulus, 0)]

                count = number_of_bins_in_stimulus - 1
                for jj in xrange(0, size(total_frames_stimulus, 0) - frames_per_bin, frames_per_bin):
                    xlabel = linspace(count_timepnt, count_timepnt + frames_per_bin, frames_per_bin + 1)
                    ax2.plot(xlabel, total_frames_stimulus[jj:jj + frames_per_bin + 2, self.required_pcs],
                             linewidth=5, color=color_map_mat[count])

                    count_timepnt += frames_per_bin
                    count -= 1

                count_timepnt += frames_per_bin

                # if ii != size(self.stimulus_off_time) - 1:
                count = number_of_bins_in_stimulus - 1
                for jj in xrange(0, size(total_frames_stimulus_off, 0) - frames_per_bin, frames_per_bin):
                    xlabel = linspace(count_timepnt, count_timepnt + frames_per_bin, frames_per_bin + 1)
                    ax2.plot(xlabel, total_frames_stimulus_off[jj:jj + frames_per_bin + 2, self.required_pcs],
                             linestyle='--',
                             linewidth=5, color=color_map_mat[count])
                    count_timepnt += frames_per_bin
                    count -= 1

                count_timepnt += 1

            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            self.plot_stimulus_patch(ax2)
            ax2.set_xlim(0, size(self.pca_components, 0))
            self.convert_frames_to_sec(fig1, ax2)
            ax2.set_xlabel("Time (seconds)")
            plt.axhline(y=0, linestyle='-', color='k', linewidth=1)

    def convert_frames_to_sec(self, fig1, ax1):

        n = size(self.pca_components, 0)
        t = 1.0 / self.frames_per_sec
        xlabels_time = linspace(0, n * t, n + 1)
        fig1.canvas.draw()
        labels = [item.get_text() for item in ax1.get_xticklabels()]
        print labels
        labels_new = [str(around(xlabels_time[int(item)])).rstrip('0').rstrip('.') for item in labels if
                      item != '']
        ax1.set_xticklabels(labels_new)

    def plot_scores(self, fig1, gs, plot_title='Habenula', gridspecs='[0,0]'):
        with sns.axes_style('white'):

            for ind in range(0, size(self.unique_clrs, 0)):
                ax1 = eval('fig1.add_subplot(gs' + '[' + str(ind) + gridspecs[-4:] + ')')

                if len(self.matched_signals) == 3:
                    plt.plot(mean(self.matched_signals[ind, :, :], axis=0), linewidth=5, color=self.unique_clrs[ind])
                else:
                    plt.plot(self.matched_signals[ind, 0, :], linewidth=5, color=self.unique_clrs[ind])

                plt.axhline(y=0, linestyle='-', color='k', linewidth=1)
                plt.xlim(0, size(self.pca_components, 0))
                plt.ylim(min(self.matched_signals), max(self.matched_signals))
                self.plot_vertical_lines_onset()
                self.plot_vertical_lines_offset()

                if ind == 0:
                    self.plot_stimulus_patch(ax1)
                    plt.axis('off')
                    plt.title(plot_title, fontsize=14)

                elif ind == size(self.unique_clrs, 0) - 1:
                    plt.xlabel("Time (seconds)")
                    plt.grid('off')
                    ax1.locator_params(axis='y', nbins=3)
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['bottom'].set_visible(False)
                    ax1.spines['left'].set_visible(False)
                    self.convert_frames_to_sec(fig1, ax1)

                else:
                    plt.axis('off')

            plt.subplots_adjust(wspace=None, hspace=None)

    def plot_pca_in_2d(self, fig1, gs, gridspecs='[0,0]'):
        with sns.axes_style('darkgrid'):
            ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')

            ax1.plot(self.pca_components[0:self.stimulus_on_time[0], self.required_pcs[0]],
                     self.pca_components[0:self.stimulus_on_time[0], self.required_pcs[1]],
                     color='#808080', linewidth=3)

            for ii in xrange(0, size(self.stimulus_on_time)):
                ax1.plot(
                    self.pca_components[self.stimulus_on_time[ii]:self.stimulus_off_time[ii], self.required_pcs[0]],
                    self.pca_components[self.stimulus_on_time[ii]:self.stimulus_off_time[ii], self.required_pcs[1]],
                    color=self.color_mat[ii], linewidth=3)

            # Plot light off times
            for ii in xrange(0, size(self.stimulus_on_time)):
                if ii != size(self.stimulus_on_time) - 1:
                    ax1.plot(self.pca_components[self.stimulus_off_time[ii]:self.stimulus_on_time[ii + 1],
                             self.required_pcs[0]],
                             self.pca_components[self.stimulus_off_time[ii]:self.stimulus_on_time[ii + 1],
                             self.required_pcs[1]],
                             color=self.color_mat[ii], linewidth=2, linestyle='--')

            ax1.set_xlabel('PC' + str(self.required_pcs[0]), linespacing=10)
            ax1.set_ylabel('PC' + str(self.required_pcs[1]), linespacing=10)

    def plot_pca_in_3d(self, fig1, gs, z_direction, gridspecs='[0,0]'):
        with sns.axes_style('darkgrid'):
            ax1 = eval('fig1.add_subplot(gs' + gridspecs + ', projection="3d")')

            # Plot Baseline at beginning
            ax1.plot(self.pca_components[0:self.stimulus_on_time[0], self.required_pcs[0]],
                     self.pca_components[0:self.stimulus_on_time[0], self.required_pcs[1]],
                     self.pca_components[0:self.stimulus_on_time[0], self.required_pcs[2]], zdir=z_direction,
                     color='#808080',
                     linewidth=2)

            # Plot light on times
            for ii in xrange(0, size(self.stimulus_on_time)):
                ax1.plot(
                    self.pca_components[self.stimulus_on_time[ii]:self.stimulus_off_time[ii], self.required_pcs[0]],
                    self.pca_components[self.stimulus_on_time[ii]:self.stimulus_off_time[ii], self.required_pcs[1]],
                    self.pca_components[self.stimulus_on_time[ii]:self.stimulus_off_time[ii], self.required_pcs[2]],
                    zdir=z_direction, color=self.color_mat[ii], linewidth=2)

            # Plot light off times
            for ii in xrange(0, size(self.stimulus_on_time)):
                if ii != size(self.stimulus_on_time) - 1:
                    ax1.plot(self.pca_components[self.stimulus_off_time[ii]:self.stimulus_on_time[ii + 1],
                             self.required_pcs[0]],
                             self.pca_components[self.stimulus_off_time[ii]:self.stimulus_on_time[ii + 1],
                             self.required_pcs[1]],
                             self.pca_components[self.stimulus_off_time[ii]:self.stimulus_on_time[ii + 1],
                             self.required_pcs[2]],
                             zdir=z_direction,
                             color=self.color_mat[ii], linewidth=2, linestyle='--')

            self.plot_axis_labels_byzdir(ax1, z_direction, self.required_pcs)

            ax1.locator_params(axis='x', pad=100, nbins=2)
            ax1.locator_params(axis='y', pad=100, nbins=2)
            ax1.locator_params(axis='z', pad=100, nbins=2)
            ax1.set_ylim((min(self.pca_components), max(self.pca_components)))
            ax1.set_xlim((min(self.pca_components), max(self.pca_components)))
            ax1.set_zlim((min(self.pca_components), max(self.pca_components)))

    @staticmethod
    def plot_axis_labels_byzdir(ax1, z_direction, required_pcs):
        if z_direction == 'y':
            ax1.set_xlabel('PC' + str(required_pcs[0]), linespacing=10)
            ax1.set_ylabel('PC' + str(required_pcs[2]), linespacing=10)

            ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax1.set_zlabel('PC' + str(required_pcs[1]), rotation=90, linespacing=10)

        elif z_direction == 'z':
            ax1.set_xlabel('PC' + str(required_pcs[0]), linespacing=10)
            ax1.set_ylabel('PC' + str(required_pcs[1]), linespacing=10)

            ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax1.set_zlabel('PC' + str(required_pcs[2]), rotation=90, linespacing=10)

        elif z_direction == 'x':
            ax1.set_xlabel('PC' + str(required_pcs[1]), linespacing=10)
            ax1.set_ylabel('PC' + str(required_pcs[2]), linespacing=10)

            ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax1.set_zlabel('PC' + str(required_pcs[0]), rotation=90, linespacing=10)

    def plotimageplanes(self, fig1, gs, gridspecs='[0, 0]'):

        if len(shape(self.maps)) == 4:
            gd_new = gridspecs
            for ii in xrange(0, size(self.maps, 2)):
                ax1 = eval('fig1.add_subplot(gs' + gd_new + ')')
                plt.imshow(self.maps[:, :, ii, :])
                # plt.title('Plane ' + str(ii), fontsize=14)
                plt.axis('off')
                print gd_new
                gd_new = gd_new[0] + str(int(gd_new[1]) + 2) + gd_new[2:]

        elif len(shape(self.maps)) == 3:
            ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
            plt.imshow(self.maps, cmap='hsv')
            plt.axis('off')
            # plt.title('Plane 1', fontsize=14)

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
            elif self.stimulus_train[ii] == "Green":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim,
                                          y_tick_width / 4, fc='green')
                ax1.add_patch(rectangle)

    def plot_stimulus_palette(self, fig1, gs, gridspecs='[0, 0]'):
        color_mat = self.color_mat
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        n = len(color_mat)
        ax1.imshow(arange(n).reshape(1, n), cmap=mpl.colors.ListedColormap(list(color_mat)), interpolation='nearest',
                   aspect='auto')
        ax1.set_xticks(arange(n) - .5)
        ax1.set_yticks([-.5, .5])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    def plot_scores_for_comparison(self, fig1, gs, scores_to_be_plotted, gridspecs='[0,0]'):

        with sns.axes_style('dark'):
            ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')

            for ind in scores_to_be_plotted:
                if len(self.matched_signals) == 3:
                    ax1.plot(mean(self.matched_signals[ind, :, :], axis=0), linewidth=3,
                             color=self.unique_clrs[ind])
                else:
                    ax1.plot(self.matched_signals[ind, 0, :], linewidth=3,
                             color=self.unique_clrs[ind])
            ax1.locator_params(axis='y', nbins=4)
            plt.axhline(y=0, linestyle='-', color='k', linewidth=1)
            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            y_lim = ax1.get_ylim()
            ax1.set_ylim((y_lim[0], y_lim[1]))
            ax1.set_xlim(0, size(self.pca_components, 0))
            self.convert_frames_to_sec(fig1, ax1)
            # self.plot_stimulus_patch(ax1)

    def plot_stimulus_patch_withtimebin(self, fig1, gs, bins=1, colors='winter', gridspecs='[0,0]', stimtype='ON'):

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
                ax2.plot(tile(total_frames_stimulus[jj], 5), linspace(0, 5, 5), '.', color=color_map_mat[count],
                         markersize=10)
                count -= 1
        plt.xlim(0, size(self.pca_components, 0))
        plt.axis('off')
