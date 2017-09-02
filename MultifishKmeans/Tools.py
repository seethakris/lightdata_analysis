import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

almost_black = '#262626'
light_grey = np.array([float(248) / float(255)] * 3)


class PlottingTools(object):
    @classmethod
    def plot_vertical_lines_onset(cls, stimulus_on_time, **kwargs):
        if 'axis_handle' in kwargs:
            ax = kwargs['axis_handle']
            for ii in xrange(0, np.size(stimulus_on_time)):
                ax.axvline(x=stimulus_on_time[ii], linestyle='-', color='k', linewidth=2)
        else:
            for ii in xrange(0, np.size(stimulus_on_time)):
                plt.axvline(x=stimulus_on_time[ii], linestyle='-', color='k', linewidth=2)

    @classmethod
    def plot_vertical_lines_offset(cls, stimulus_off_time, **kwargs):
        if 'axis_handle' in kwargs:
            ax = kwargs['axis_handle']
            for ii in xrange(0, np.size(stimulus_off_time)):
                ax.axvline(x=stimulus_off_time[ii], linestyle='--', color='k', linewidth=2)
        else:
            for ii in xrange(0, np.size(stimulus_off_time)):
                plt.axvline(x=stimulus_off_time[ii], linestyle='--', color='k', linewidth=2)

    @classmethod
    def plot_stimulus_patch(cls, stimulus_on_time, stimulus_off_time, stimulus_train, axis_handle):
        # Get y grid size
        y_tick = axis_handle.get_yticks()
        y_tick_width = y_tick[2] - y_tick[1]

        # Adjust axis to draw patch
        y_lim = axis_handle.get_ylim()
        axis_handle.set_ylim((y_lim[0], y_lim[1] + y_tick_width / 3))

        for ii in xrange(0, np.size(stimulus_on_time)):
            # Find time of stimulus for width of patch
            time_stim = stimulus_off_time[ii] - stimulus_on_time[ii]
            # Check different cases of stimuli to create patches
            if stimulus_train[ii] == "Blue":
                rectangle = plt.Rectangle((stimulus_on_time[ii], y_lim[1]), time_stim,
                                          y_tick_width / 4, fc='blue')
                axis_handle.add_patch(rectangle)
            elif stimulus_train[ii] == "Red":
                rectangle = plt.Rectangle((stimulus_on_time[ii], y_lim[1]), time_stim,
                                          y_tick_width / 4, fc='red')
                axis_handle.add_patch(rectangle)
            elif stimulus_train[ii] == "UV":
                rectangle = plt.Rectangle((stimulus_on_time[ii], y_lim[1]), time_stim,
                                          y_tick_width / 4, fc='purple')
                axis_handle.add_patch(rectangle)

    @classmethod
    def smooth_hanning(cls, x, window_len, window='hanning'):
        s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = np.hanning(window_len)
        y = np.convolve(w / w.sum(), s, mode='valid')
        return y[:-window_len + 1]

    @classmethod
    def format_legend(cls, axis_handle, **kwargs):

        if 'legend_location' in kwargs:
            legend_location = kwargs['legend_location']
            bbox = (0.5, 0)
        else:
            legend_location = 'center left'
            bbox = (1, 0.5)

        if 'legend_string' in kwargs:
            legend_string = kwargs['legend_string']
            legend = axis_handle.legend(legend_string, frameon=True, scatterpoints=1, loc=legend_location,
                                        bbox_to_anchor=bbox)
        else:
            legend = axis_handle.legend(frameon=True, scatterpoints=1, loc=legend_location, bbox_to_anchor=bbox)

        # Shrink current axis by 20%
        box = axis_handle.get_position()
        axis_handle.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        rect = legend.get_frame()
        rect.set_facecolor(light_grey)
        rect.set_linewidth(0.0)

        # Change the legend label colors to almost black, too
        texts = legend.texts
        for t in texts:
            t.set_color(almost_black)

    @classmethod
    def format_axis(cls, axis_handle, **kwargs):
        if 'title' in kwargs:
            axis_handle.set_title(kwargs['title'])
        if 'xlabel' in kwargs:
            axis_handle.set_xlabel(kwargs['xlabel'])
            axis_handle.locator_params(axis='x', pad=50, nbins=8)
        if 'ylabel' in kwargs:
            axis_handle.set_ylabel(kwargs['ylabel'])
            axis_handle.locator_params(axis='y', pad=50, nbins=4)
        if 'zlabel' in kwargs:
            axis_handle.set_zlabel(kwargs['zlabel'])
            axis_handle.locator_params(axis='z', pad=50, nbins=4)
        if 'zeroline' in kwargs:
            if kwargs['zeroline']:
                axis_handle.axhline(y=0, linestyle='-', color='k', linewidth=1)
        if 'xlim' in kwargs:
            axis_handle.set_xlim((kwargs['xlim'][0], kwargs['xlim'][1]))
        if 'ylim' in kwargs:
            axis_handle.set_ylim((kwargs['ylim'][0], kwargs['ylim'][1]))

            # plt.tight_layout()


class PCAPlottingTools(object):
    @classmethod
    def plot_scores_in2d(cls, ax1, pca_scores, stimulus_on_time, stimulus_off_time, color_mat):

        ax1.plot(pca_scores[0:stimulus_on_time[0], 0],
                 pca_scores[0:stimulus_on_time[0], 1],
                 color='#808080', linewidth=3)

        for ii in xrange(0, np.size(stimulus_on_time)):
            ax1.plot(pca_scores[stimulus_on_time[ii]:stimulus_off_time[ii], 0],
                     pca_scores[stimulus_on_time[ii]:stimulus_off_time[ii], 1],
                     color=color_mat[ii], linewidth=3)

        ax1.plot(pca_scores[stimulus_off_time[-1] + 20:, 0],
                 pca_scores[stimulus_off_time[-1] + 20:, 1],
                 color='#000000', linewidth=3)

        # Plot light off times
        for ii in xrange(0, np.size(stimulus_on_time)):
            if ii == np.size(stimulus_on_time) - 1:
                ax1.plot(
                    pca_scores[stimulus_off_time[ii]:stimulus_off_time[ii] + 20, 0],
                    pca_scores[stimulus_off_time[ii]:stimulus_off_time[ii] + 20, 1],
                    color=color_mat[ii], linewidth=2, linestyle='--')
            else:
                ax1.plot(pca_scores[stimulus_off_time[ii]:stimulus_on_time[ii + 1], 0],
                         pca_scores[stimulus_off_time[ii]:stimulus_on_time[ii + 1], 1],
                         color=color_mat[ii], linewidth=2, linestyle='--')

    @classmethod
    def plot_scores_in3d(cls, ax1, pca_scores, stimulus_on_time, stimulus_off_time, stimulus_types,
                         color_mat):
        # Plot Baseline at beginning
        ax1.plot(pca_scores[0:stimulus_on_time[0], 0],
                 pca_scores[0:stimulus_on_time[0], 1],
                 pca_scores[0:stimulus_on_time[0], 2], zdir='z', color='#808080',
                 linewidth=2, label='Baseline1')

        # Plot light on times
        for ii in xrange(0, np.size(stimulus_on_time)):
            ax1.plot(pca_scores[stimulus_on_time[ii]:stimulus_off_time[ii], 0],
                     pca_scores[stimulus_on_time[ii]:stimulus_off_time[ii], 1],
                     pca_scores[stimulus_on_time[ii]:stimulus_off_time[ii], 2],
                     zdir='z', color=color_mat[ii], linewidth=2, label=stimulus_types[ii] + ' ON')

        ## Plot Baseline at end of stimulus
        ax1.plot(pca_scores[stimulus_off_time[ii] + 20:, 0],
                 pca_scores[stimulus_off_time[ii] + 20:, 1],
                 pca_scores[stimulus_off_time[ii] + 20:, 2], zdir='z',
                 color='#000000', linewidth=2, label='Baseline2')

        # Plot light off times
        for ii in xrange(0, np.size(stimulus_on_time)):
            if ii == np.size(stimulus_on_time) - 1:
                ax1.plot(
                    pca_scores[stimulus_off_time[ii]:stimulus_off_time[ii] + 20, 0],
                    pca_scores[stimulus_off_time[ii]:stimulus_off_time[ii] + 20, 1],
                    pca_scores[stimulus_off_time[ii]:stimulus_off_time[ii] + 20, 2],
                    zdir='z', color=color_mat[ii], linewidth=2, linestyle='--', label=stimulus_types[ii] + ' OFF')
            else:

                ax1.plot(pca_scores[stimulus_off_time[ii]:stimulus_on_time[ii + 1], 0],
                         pca_scores[stimulus_off_time[ii]:stimulus_on_time[ii + 1], 1],
                         pca_scores[stimulus_off_time[ii]:stimulus_on_time[ii + 1], 2],
                         zdir='z',
                         color=color_mat[ii], linewidth=2, linestyle='--', label=stimulus_types[ii] + ' OFF')

        PlottingTools.format_legend(axis_handle=ax1)

    @classmethod
    def format_axis(cls, axis_handle, **kwargs):
        if 'title' in kwargs:
            axis_handle.set_title(kwargs['title'])
        if 'xlabel' in kwargs:
            axis_handle.set_xlabel(kwargs['xlabel'])
            axis_handle.locator_params(axis='x', pad=100, nbins=4)
        if 'ylabel' in kwargs:
            axis_handle.set_ylabel(kwargs['ylabel'])
            axis_handle.locator_params(axis='y', pad=100, nbins=4)
        if 'zlabel' in kwargs:
            axis_handle.set_zlabel(kwargs['zlabel'])
            axis_handle.locator_params(axis='z', pad=100, nbins=4)
        if 'xlim' in kwargs:
            axis_handle.set_xlim((kwargs['xlim'][0], kwargs['xlim'][1]))
        if 'ylim' in kwargs:
            axis_handle.set_ylim((kwargs['ylim'][0], kwargs['ylim'][1]))
        if 'zlim' in kwargs:
            axis_handle.set_ylim((kwargs['ylim'][0], kwargs['ylim'][1]))

        plt.tight_layout()
