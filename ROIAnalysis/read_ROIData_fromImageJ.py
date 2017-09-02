from numpy import r_, ones, convolve
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
import pickle
from matplotlib.backends.backend_pdf import PdfPages

filesep = os.path.sep


class CollectandPlot(object):
    def __init__(self, FolderName, framerate, stimulus_on_time, stimulus_off_time, stimulus_train, time_end, startframe, baseline,
                 smooth_window, figureflag=False):
        self.FolderName = FolderName

        self.savefigurefolder = os.path.join(self.FolderName, 'Figures') + filesep
        self.makedirs(self.savefigurefolder)
        self.savedatafolder = os.path.join(self.FolderName, 'PickledData') + filesep
        self.makedirs(self.savedatafolder)

        self.startframe = startframe
        self.smooth_window = smooth_window
        self.time_end = time_end

        self.baseline = [ii - self.startframe for ii in baseline]
        self.stimulus_on_time = [ii - self.startframe for ii in stimulus_on_time]
        self.stimulus_off_time = [ii - self.startframe for ii in stimulus_off_time]

        experiment_parameters = {'stimulus_on_time': self.stimulus_on_time, 'stimulus_off_time': self.stimulus_off_time,
                                 'stimulus_train': stimulus_train, 'baseline_for_dff': self.baseline,
                                 'framerate': framerate, 'smooth_window': smooth_window, 'frame_start': self.startframe}

        # Get number of folders in file
        self.Fishname = [ii for ii in os.listdir(self.FolderName) if ii.find('Fish') == 0]

        ## Sort and save data
        data_blue = []
        centroid = []
        shape_x_dict = {}
        for fish in self.Fishname:
            Filenames = [f for f in os.listdir(os.path.join(self.FolderName, fish)) if
                         (f.endswith('.xls') or f.endswith('.csv'))]
            for file in Filenames:
                print 'Analysing ...%s, file %s' % (fish, file)
                currentfilepath = os.path.join(self.FolderName, fish, file)
                if not os.path.exists(currentfilepath[:-4] + '.csv'):
                    os.rename(currentfilepath, currentfilepath[:-4] + '.csv')
                # Data folders
                if file.find('Centroid') == 0:
                    temp, shape_x = self.get_centroid_info(fish, file, currentfilepath)
                    shape_x_dict[temp['fish']] = shape_x
                    centroid.append(temp)
                else:

                    temp = self.collectandsortdata(fish, file, currentfilepath, shape_x_dict)
                    data_blue.append(temp)
        if figureflag:
            pp = PdfPages(os.path.join(self.savefigurefolder, 'Blue.pdf'))
            self.run_through_data_and_plot(data_blue, pp)
            pp.close()

        self.pickledata(data_blue, 'data_blue')
        self.pickledata(centroid, 'centroid')
        self.pickledata(experiment_parameters, 'experiment_parameters')

    def run_through_data_and_plot(self, data, pp):
        for ii in xrange(0, np.size(data)):

            fishname = data[ii]['fish']
            count = 0
            fs = plt.figure(figsize=(25, 10))
            gs = plt.GridSpec(2, np.size(data[ii].keys()) - 1)

            for ind, value in data[ii].iteritems():
                if ind == 'fish' or ind == 'raw_data':
                    continue
                elif ind == 'detrend_data':
                    vmin = 0
                    vmax = 50
                else:  # If it is normalized or smoothed
                    vmin = -0.5
                    vmax = 0.5

                ax1 = fs.add_subplot(gs[0, count])
                self.plot_image(value, ind, vmin, vmax, fishname)
                ax1 = fs.add_subplot(gs[1, count])

                plt.plot(value, color='grey', alpha=0.5)
                plt.plot(np.mean(value, 1), color='black', linewidth=2)
                count += 1

            pp.savefig(fs, bbox_inches='tight')
            plt.close()

    def plot_image(self, data, datalabel, vmin, vmax, fishname):
        plt.imshow(np.sort(data, axis=1).T, cmap='jet', interpolation='nearest', aspect='auto', vmin=vmin,
                   vmax=vmax)

        plt.grid('off')
        plt.title(fishname + '' + datalabel)
        self.plot_stimulus_lines_onset(linewidth=1)
        plt.locator_params(axis='y', nbins=6)
        plt.xlim([0, np.size(data, 0)])
        plt.ylim([np.size(data, 1) - 1, 0])
        plt.colorbar()

    def collectandsortdata(self, fish, filename, currentfilepath, shape_x):
        dataframe = pd.read_csv(currentfilepath[:-4] + '.csv', sep='\t')
        data = self.process_data(fish, filename, dataframe, shape_x)
        return data

    def process_data(self, fish, filename, dataframe, shape_x):
        data = dataframe.values[self.startframe:time_end, 1:-3]
        # data = data[:, :shape_x[fish + '_' + filename[:-4]]]

        # print 'Blas', shape_x[fish + '_' + filename[:-4]], np.shape(data)
        detrend_data = np.zeros(np.shape(data))
        for ii in xrange(0, np.size(data, 1)):
            detrend_data[:, ii] = detrend(data[:, ii])

        zscore_data = (detrend_data - np.mean(detrend_data, 0)) / (np.std(detrend_data, 0) + 0.001)
        smooth_data_zs = np.zeros(np.shape(zscore_data))
        deltaff_data = np.zeros(np.shape(zscore_data))
        smooth_data_dff = np.zeros(np.shape(zscore_data))

        for ii in xrange(0, np.size(zscore_data, 1)):
            deltaff_data[:, ii] = (data[:, ii] - np.mean(data[self.baseline[0]:self.baseline[1], ii])) / (
                abs(np.mean(data[self.baseline[0]:self.baseline[1], ii])) + 0.00001)
            smooth_data_zs[:, ii] = self.smooth_func(zscore_data[:, ii], self.smooth_window)
            smooth_data_dff[:, ii] = self.smooth_func(deltaff_data[:, ii], self.smooth_window)

        print 'Data ', np.shape(smooth_data_dff)
        # Plot heatmaps and save
        return {'fish': fish + '_' + filename[:-4], 'raw_data': data, 'detrend_data': detrend_data,
                'zscore_data': zscore_data, 'deltaff_data': deltaff_data, 'smooth_zs': smooth_data_zs,
                'smooth_dff': smooth_data_dff}

    def get_centroid_info(self, fishname, filename, currentfilepath):
        centroid = pd.read_csv(currentfilepath[:-4] + '.csv', sep='\t')
        x = centroid['X']
        minimum = np.min(x)
        x -= minimum
        y = centroid['Y']
        print 'Centroid ', np.shape(x)
        # area = centroid['Area']
        return {'fish': fishname + '_' + filename[9:-4], 'x': x, 'y': y}, np.size(x)

    ## Smooth and stimulus onset and offset
    @staticmethod
    def smooth_func(x, window_len=10):
        s = r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        w = ones(window_len, 'd')
        y = convolve(w / w.sum(), s, mode='valid')
        return y[window_len / 2:-window_len / 2 + 1]

    def plot_stimulus_lines_onset(self, linewidth=2):
        for ii in xrange(0, np.size(self.stimulus_on_time)):
            plt.axvline(x=self.stimulus_on_time[ii], linestyle='-', color='k', linewidth=linewidth)
            plt.axvline(x=self.stimulus_off_time[ii], linestyle='--', color='k', linewidth=linewidth)

    @staticmethod
    def makedirs(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def pickledata(self, data, savename):
        with open(os.path.join(self.savedatafolder, savename), 'wb') as fp:
            pickle.dump(data, fp)


if __name__ == '__main__':
    Folder = '/Users/seetha/Google Drive/BMC_Revised_manuscript/Thalamus/AF4Lesion/Tiff/After/'

    # Blue x 7
    # stimulus_on_time = [244, 764, 1284, 1804, 2324, 2844, 3369]
    # stimulus_off_time = [504, 1024, 1544, 2064, 2584, 3104, 3629]

    # Bluex4 - 13fps
    # stimulus_on_time = [588, 1105, 164, 2142]
    # stimulus_off_time = [847, 1365, 1900, 2418]

    # # Bluex4 - 7 fps
    # stimulus_on_time = [280, 543, 814, 1069]
    # stimulus_off_time = [408, 671, 942, 1213]

    #4 pulses
    stimulus_on_time = [46, 86, 126, 166]
    stimulus_off_time = [65, 105, 145, 185]

    # # BlueRed x3
    # stimulus_on_time = [43, 63, 83, 150, 170, 190]
    # stimulus_off_time = [53, 73, 93, 160, 180, 200]

    # #BlueRedx3
    # stimulus_on_time = [46, 86, 126, 166, 206, 246]
    # stimulus_off_time = [65, 105, 145, 185, 225, 265]

    # stimulus_on_time = [272, 513, 755, 1000, 1237, 1483]
    # stimulus_off_time = [414, 655, 893, 1138, 1383, 1624]
    framerate = 1

    stimulus_train = ['Blue', 'Blue', 'Blue', 'Blue']
    smooth_window = 3
    frame_start = 10
    baseline = [10, 40]
    time_end = 205
    figureflag = False
    CollectandPlot(Folder, framerate, stimulus_on_time, stimulus_off_time, stimulus_train, time_end, frame_start, baseline,
                   smooth_window, figureflag)

