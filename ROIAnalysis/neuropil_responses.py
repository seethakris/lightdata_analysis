import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from numpy import r_, ones, convolve
import scipy as sp
import scipy.stats
import pandas as pd
import peakutils
from itertools import chain
from scipy.signal import detrend
import matplotlib

sns.set_context('paper', font_scale=1.3)


class PlotNeuropil(object):
    def __init__(self, ExperimentFolder, img_size, time, numberofplanes, fastONtracelength, smooth_window, ON_corr,
                 OFF_corr, Both_corr, neuropilpixelthresh, heatmap_values):
        self.ExperimentFolder = ExperimentFolder
        self.smooth_window = smooth_window
        self.time = time
        self.img_size = img_size
        self.fastONtracelength = fastONtracelength
        self.neuropilpixelthresh = neuropilpixelthresh

        self.corr_threshold = {'fastON': ON_corr, 'slowON': ON_corr, 'OFF': OFF_corr, 'Both': Both_corr}
        self.colors = {'Both': 'yellow', 'fastON': 'blue', 'slowON': 'cyan', 'Inh': 'green', 'OFF': 'red'}
        self.numberofplanes = numberofplanes

        ## Save data
        self.ResultFolder = os.path.join(ExperimentFolder, 'SaveResult')
        if not os.path.exists(self.ResultFolder):
            os.makedirs(self.ResultFolder)

        self.PickleFolder = os.path.join(ExperimentFolder, 'PickledData')
        with open(os.path.join(self.PickleFolder, 'experiment_parameters')) as fp:
            ExperimentParameters = pickle.load(fp)
            self.stimulus_on_time = ExperimentParameters['stimulus_on_time']
            self.stimulus_off_time = ExperimentParameters['stimulus_off_time']

            self.baseline = ExperimentParameters['baseline_for_dff']
            self.framerate = ExperimentParameters['framerate']
            self.startframe = ExperimentParameters['frame_start']

        self.stimulus_on_time_ins = [np.round(ii / self.framerate) for ii in self.stimulus_on_time]
        self.stimulus_off_time_ins = [np.round(ii / self.framerate) for ii in self.stimulus_off_time]
        self.ITI = self.stimulus_off_time[0] - self.stimulus_on_time[0]

        self.dummytraces = self.create_dummy_trace()
        self.Fishname = [ii for ii in os.listdir(self.ExperimentFolder) if ii.find('Fish') == 0]

        # Open tiff files
        # Combine and plot
        fs = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, self.numberofplanes, height_ratios=[1, 2, 2, 2])
        self.numcells = pd.DataFrame(columns=['Fish', 'Numcells', 'Plane', 'ResponseType', 'Type'])
        for planes in xrange(1, self.numberofplanes + 1):
            self.OnOffmatrix = []
            self.imagesize = []
            self.neuropil = []
            self.responsetime = []
            self.OnOfftrace = []
            # self.Ontimetrace = []
            # self.Offtimetrace = []
            for fish in self.Fishname:
                ImageFolder = os.path.join(self.ExperimentFolder, fish, 'ImagesUsed')
                tifffiles = [ii for ii in os.listdir(ImageFolder) if ii.find('Neuropil') == 0]
                for tiff in tifffiles:
                    if tiff.find('Plane' + str(planes)) > 0:
                        print 'Plane %s, Fish %s, Tiff %s' % (planes, fish, tiff)
                        # axisON = plt.subplot(gs[3, planes - 1])
                        temp1, temp2, temp3, temp4, = self.open_tifffiles(fish, os.path.join(ImageFolder, tiff))
                        # temp1, temp2, temp3, temp4, temp5, temp6 = self.open_tifffiles(axisON,  fish,
                        #                                                                os.path.join(ImageFolder, tiff))

                        # ONOFFmatrix= Boolean array with cells classified using correlation = true
                        # ONOFFtrace = stacked time series data by classification
                        # responsetime = time of response by classification
                        # neuropil = pixels belonging to neuropil
                        self.OnOffmatrix.append(temp1)
                        self.OnOfftrace.append(temp2)
                        self.neuropil.append(temp3)
                        self.responsetime.append(temp4)
                        # self.Ontimetrace.append(temp5)
                        # self.Offtimetrace.append(temp6)
                        self.find_num_cells(temp1, fish, planes)

            ax1 = fs.add_subplot(gs[0, planes - 1])
            flatui = ["Yellow", "magenta", "Black", "Cyan", "Blue"]
            createdcmap = matplotlib.colors.ListedColormap(flatui)
            self.plot_location_as_heatmap(colormap=createdcmap, heatmap_values=heatmap_values)
            # self.plot_location_of_all()
            ax1 = fs.add_subplot(gs[1, planes - 1])
            self.plot_traces()
            ax1 = fs.add_subplot(gs[2, planes - 1])
            self.responsetime_all = self.plot_cdf()
            # ax1 = fs.add_subplot(gs[3, planes - 1])
            # self.plot_num_cells(fish, planes)

        # fs.savefig(os.path.join(self.ResultFolder, 'Neuropilfigure2.pdf'), bbox_inches='tight')

        # np.savez(os.path.join(self.ResultFolder, 'Neuropilresults.npz'), OnOffmatrix=self.OnOffmatrix,
        #          OnOfftrace=self.OnOfftrace, neuropil=self.neuropil,
        #          responsetime=self.responsetime, responsetime_all=self.responsetime_all, numcells=self.numcells)

    def open_tifffiles(self, fish, tiff):
        img = Image.open(tiff)
        imgArray = np.zeros((self.img_size[0], self.img_size[1], self.time), dtype=np.uint16)
        for tt in range(self.time):
            img.seek(tt)
            imgArray[:, :, tt] = img
        imgArray = imgArray[:, :, self.startframe:]
        self.imagesize = np.shape(imgArray)[0:2]

        # Use only pixels that belong to neuropil to analyse
        neuropil = np.where(np.mean(imgArray, 2) > self.neuropilpixelthresh)
        print 'Pixels belonging to neuropil', np.shape(neuropil[0])
        # Correlate pixels
        # Response_traces, Ontrace, Offtrace = self.correlation_coefficient(fish, imgArray, neuropil, axisOn=ax1)
        Response_traces = self.correlation_coefficient(fish, imgArray, neuropil)
        Classified_pixeltraces = self.plot_location_of_pixels(imgArray, neuropil, Response_traces)

        time_responses = self.find_time_of_response(Classified_pixeltraces)
        # self.plot_mean_of_on_and_off(fig, grid, Classified_pixeltraces)

        return Response_traces, Classified_pixeltraces, neuropil, time_responses  # , Ontrace, Offtrace

    def find_num_cells(self, OnOffmatrix, fish, plane):
        # # Find number of pixels in the neuropil
        # dict = {key: [] for key in self.OnOffmatrix[0].iterkeys()}
        # for ii in xrange(0, np.size(self.OnOffmatrix)):
        #     for keys, values in self.OnOffmatrix[ii].iteritems():
        #         number_of_pixels = (np.size(np.where(values == 1)) / float(np.size(values))) * 100
        #         dict[keys].append(number_of_pixels)

        for keys, values in OnOffmatrix.iteritems():
            numberofcells = (np.size(np.where(np.asarray(values) == 1)) / float(np.size(values, 0))) * 100
            # if keys.find('ON') >= 0:
            #     responsetype = keys
            # elif keys.find('OFF') or keys.find('Both'):
            #     responsetype = 'OFF'
            # else:
            #     continue

            temp = pd.DataFrame([[fish, numberofcells, plane, keys, 'Neuropil']], columns=self.numcells.columns)
            self.numcells = self.numcells.append(temp, ignore_index=True)

            # return dict

    def plot_num_cells(self, fish, plane):
        # Plot percentage of cells in left and right for on and off
        numcells = self.find_num_cells(fish, plane)
        OnOff = pd.DataFrame.from_dict(numcells)
        sns.boxplot(data=OnOff, linewidth=2)
        sns.stripplot(data=OnOff, jitter=True, linewidth=1, alpha=0.5, size=6)

        ## Save as csv
        OnOff.to_csv(os.path.join(self.ResultFolder, 'NeuropilOnOff_percentage.csv'))
        return numcells

    def plot_traces(self):

        for keys in self.OnOfftrace[0].iterkeys():
            templist = np.array([])
            if keys != 'NoResponse':
                for ii in xrange(0, np.size(self.OnOfftrace)):
                    times = np.asarray(self.OnOfftrace[ii][keys])
                    if np.size(times) != 0:
                        templist = np.vstack((templist, times)) if templist.size else times

                # Mean and confidence interval
                if np.size(templist) != 0:
                    templist = templist.T
                    label_in_seconds = self.convert_frames_to_seconds(np.size(templist, 0))[:-1]
                    y, error1, error2 = self.mean_confidence_interval(templist)
                    # print np.shape(label_in_seconds), np.shape(y), np.shape(error1)
                    plt.plot(label_in_seconds, y, label=keys, color=self.colors[keys], linewidth=1)
                    plt.fill_between(label_in_seconds, error1, error2, alpha=0.5)
        self.plot_stimulus_lines_onset(secondsflag=True)
        plt.grid('off')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Delta f/f')
        plt.legend()

    def plot_location_of_all(self):
        for count in xrange(0, np.size(self.neuropil, 0)):

            neuropillist = self.neuropil[count]
            pixels = np.size(neuropillist, 1)
            # for ii in xrange(0, pixels):
            #     plt.plot(neuropillist[1][ii], neuropillist[0][ii], 'k.')

            for keys, values in self.OnOffmatrix[count].iteritems():
                for ii in xrange(0, pixels):
                    if values[ii]:
                        plt.plot(neuropillist[1][ii], neuropillist[0][ii], '.', markersize=3, color=self.colors[keys],
                                 alpha=0.5)

        # plt.xlim((20, 210))
        plt.ylim((0, self.img_size[0]))
        plt.axis('off')
        plt.gca().invert_yaxis()

    def plot_location_as_heatmap(self, colormap, heatmap_values):
        # Count number of pixels and make heatmap
        imagematrix = np.zeros(self.imagesize)
        for count in xrange(0, np.size(self.neuropil, 0)):
            neuropillist = self.neuropil[count]
            pixels = np.size(neuropillist, 1)
            for keys, values in self.OnOffmatrix[count].iteritems():
                for ii in xrange(0, pixels):
                    if values[ii]:
                        imagematrix[neuropillist[0][ii], neuropillist[1][ii]] = heatmap_values[keys]
        imagematrix = np.ma.masked_where(imagematrix == 0, imagematrix)

        cmap = plt.get_cmap(colormap)
        cmap.set_bad(color='black')

        plt.imshow(imagematrix, vmin=-2, vmax=2, cmap=cmap, interpolation='bilinear')

        # plt.colorbar()
        plt.grid('off')
        plt.axis('off')

    def plot_location_of_pixels(self, imageArray, neuropillist, Classified):
        pixels = np.size(neuropillist, 1)
        dict = {key: [] for key in Classified.iterkeys()}
        count = 0

        No_responsepixels = [1] * pixels
        # for ii in xrange(0, pixels):
        # plt.plot(neuropillist[1][ii], neuropillist[0][ii], 'k.')

        for keys, values in Classified.iteritems():
            for ii in xrange(0, pixels):
                if values[ii]:
                    No_responsepixels[ii] = 0
                    temp = imageArray[neuropillist[0][ii], neuropillist[1][ii], :]
                    temp_dff = (temp - np.mean(temp)) / (abs(np.mean(temp)) + 0.00001)
                    temp_smooth = self.smooth_func(temp_dff, window_len=self.smooth_window)
                    temp_smooth = detrend(temp_smooth)
                    dict[keys].append(temp_smooth)
                    # plt.plot(neuropillist[1][ii], neuropillist[0][ii], '.', color=self.colors[count])
            dict[keys] = np.asarray(dict[keys])
            count += 1

        pixels = np.where(np.asarray(No_responsepixels) == 1)
        dict['NoResponse'] = imageArray[neuropillist[0][pixels], neuropillist[1][pixels], :]

        # plt.xlim((20, 210))
        # plt.ylim((0, self.img_size[0]))
        # plt.axis('off')
        # plt.gca().invert_yaxis()
        # print 'Number of No Response', np.shape(dict['NoResponse'])
        return dict

    def find_time_of_response(self, Pixeltraces):
        dict = {key: [] for key in Pixeltraces.iterkeys()}
        dict.pop('NoResponse', None)

        for keys, values in Pixeltraces.iteritems():
            for ii in xrange(0, np.size(values, 0)):
                temp = values[ii, :]
                for kk in xrange(0, len(self.stimulus_on_time) - 3):
                    if keys.find('ON') >= 0:
                        trace = temp[self.stimulus_on_time[kk]:self.stimulus_off_time[kk]]
                        index = self.find_peak(trace)
                    elif keys.find('OFF') >= 0:
                        trace = temp[self.stimulus_off_time[kk]:self.stimulus_off_time[kk] + self.ITI]
                        index = self.find_peak(trace)
                    elif keys.find('Both') >= 0:
                        trace = temp[self.stimulus_on_time[kk]: self.stimulus_off_time[kk]]
                        # print np.shape(trace), np.shape(temp)
                        index1 = self.find_peak(trace)
                        trace = temp[self.stimulus_off_time[kk]:self.stimulus_off_time[kk] + self.ITI]
                        index2 = self.find_peak(trace)
                        temp1 = [list for list in chain(index1, index2) if list > 0]
                        if temp1:
                            index = [np.min(temp1)]
                        else:
                            index = [0]
                    else:  # Any other keys
                        continue
                    if len(index) == 1 and index[0] == 0:
                        continue
                    else:
                        dict[keys].append(index[0] / float(self.framerate))
        return dict

    def find_peak(self, trace):
        index = peakutils.indexes(trace, thres=1)
        # print index
        if len(index) == 0:
            return [0]
        else:
            return index

    def plot_cdf(self):
        dict = {}
        for keys in self.responsetime[0].iterkeys():
            templist = np.array([])
            for ii in xrange(0, np.size(self.responsetime)):
                times = np.asarray(self.responsetime[ii][keys])
                templist = np.hstack((templist, times)) if templist.size else times

            dict[keys] = templist
            num_bins = 20

            counts, bin_edges = np.histogram(templist, bins=num_bins, normed=True)
            cdf = np.cumsum(counts)
            plt.plot(bin_edges[1:], cdf, self.colors[keys], label=keys)
        plt.legend()
        return dict

    def plot_mean_of_on_and_off(self, fig, grid, Pixeltraces):

        count = 0
        for keys, values in Pixeltraces.iteritems():
            ax1 = fig.add_subplot(grid[count, 1])
            plt.plot(values.T, color='gray', alpha=0.5)
            plt.plot(np.mean(values, 0), color='k', linewidth=1)
            plt.title(keys)
            self.plot_stimulus_lines_onset()
            count += 1

    def create_dummy_trace(self):
        # Create dummy traces for correlation
        slowOn_trace = np.zeros(self.time - self.startframe)
        fastOn_trace = np.zeros(self.time - self.startframe)
        Off_trace = np.zeros(self.time - self.startframe)
        On_and_off_trace = np.zeros(self.time - self.startframe)

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

    def correlation_coefficient(self, fish, imageArray, neuropillist, axisOn=0):
        # Correlate each cell with the dummy traces
        pixels = np.size(neuropillist, 1)
        correlation_matrix = np.zeros((pixels, len(self.dummytraces.keys())))
        data_forhistplotting = np.array([])
        for ii in xrange(0, pixels):
            data = imageArray[neuropillist[0][ii], neuropillist[1][ii], :]
            data_dff = (data - np.mean(data)) / (abs(np.mean(data)) + 0.00001)
            data_smooth = self.smooth_func(data_dff, window_len=10)
            count = 0

            # data_forhistplotting = np.vstack(
            #     (data_forhistplotting, data_smooth)) if data_forhistplotting.size else data_smooth

            for types in self.dummytraces.itervalues():
                correlation_matrix[ii, count] = np.corrcoef(data_smooth, types)[0, 1]
                count += 1

        # plt.sca(axisOn)
        # print 'datashape', np.shape(data_forhistplotting)
        # n, bins, patches = plt.hist(
        #     np.max(data_forhistplotting[:, self.stimulus_on_time[0]:self.stimulus_off_time[0]], 1),
        #     bins=50, histtype='step', color='blue', linewidth=2, alpha=0.5)
        #
        # n, bins, patches = plt.hist(
        #     np.max(data_forhistplotting[:, self.stimulus_off_time[0]:self.stimulus_on_time[1]], 1),
        #     bins=50, histtype='step', color='magenta', linewidth=2, alpha=0.5)

        # labels = ['%2.0d' % ((item / 1300) * 100) for item in axisOn.get_yticks()]
        # axisOn.set_yticklabels(labels)
        #
        # axisOn.set_xlim((-1, 2))
        # # axisOn.set_ylim((0, 250))
        # axisOn.locator_params(axis='y', bins=4)
        # axisOn.set_xlabel('Delta f/f')
        # axisOn.set_ylabel('% of pixels')
        #
        # Ontimetrace = data_forhistplotting[:, self.stimulus_on_time[0]:self.stimulus_off_time[0]]
        # Offtimetrace = data_forhistplotting[:, self.stimulus_off_time[0]:self.stimulus_on_time[1]]

        dict_matrix = {}
        count = 0
        print 'Fish Number: %s' % fish
        for types in self.dummytraces.keys():
            dict_matrix[types] = (correlation_matrix[:, count] > self.corr_threshold[types])
            print 'Number of %s cells is %s' % (types, np.size(np.where(dict_matrix[types] == 1)))
            count += 1

        return dict_matrix  # , Ontimetrace, Offtimetrace

    @staticmethod
    def smooth_func(x, window_len=10):
        s = r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        w = ones(window_len, 'd')
        y = convolve(w / w.sum(), s, mode='valid')
        return y[window_len / 2:-window_len / 2 + 1]

    def plot_stimulus_lines_onset(self, linewidth=2, secondsflag=False):
        if secondsflag:
            for ii in xrange(0, np.size(self.stimulus_on_time_ins)):
                plt.axvline(x=self.stimulus_on_time_ins[ii], linestyle='-', color='k', linewidth=linewidth)
                plt.axvline(x=self.stimulus_off_time_ins[ii], linestyle='--', color='k', linewidth=linewidth)
        else:
            for ii in xrange(0, np.size(self.stimulus_on_time)):
                plt.axvline(x=self.stimulus_on_time[ii], linestyle='-', color='k', linewidth=linewidth)
                plt.axvline(x=self.stimulus_off_time[ii], linestyle='--', color='k', linewidth=linewidth)

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
