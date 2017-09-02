""" Preprocess Raw Data """

__author__ = "Seetha Krishnan"
__copyright__ = "Copyright (C) 2016 Seetha Krishnan"
__license__ = "Public Domain"
__version__ = "1.0"


from numpy import size, shape, mean, shape, convolve, r_, ones
import math
import matplotlib.pyplot as plt
from thunder import Colorize
import seaborn as sns
import matplotlib.patches as ptch
from thunder import Registration
import scipy
import scipy.signal
import os


class class_preprocess_data(object):
    # These functions preprocess the dataset

    def __init__(self, tsc, stimulus_on_time, stimulus_off_time, stimulus_train, saveseriesdirectory,
                 multiplane=False):
        """
        Initialize

        Parameters
        ----------
        tsc : ThunderContext
        stimulus_on_time : time of stimulus ON
        stimulus_off_time : time of stimulus OFF
        stimulus_train : Type of stimuli
        saveseriesdirectory : Where to save data as a series
        multiplane : Whether data is singleplane or multiplane. Currently only works with single plane data

        """
        self.thundercontext = tsc
        self.image = Colorize.image
        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.stimulus_train = stimulus_train
        self.saveseriesdirectory = saveseriesdirectory
        self.multiplane = multiplane

    def load_and_preprocess_data(self, FileName, crop=0, register=0, img_size_crop_x1=0, img_size_crop_x2=0,
                                 img_size_crop_y1=0, img_size_crop_y2=0, medianfilter_window=1, start_frame=0,
                                 end_frame=285, pdffile=False):

        """
        Load and preprocess data - Filter, crop and register.

        Parameters
        ----------
        FileName : Folder name for the location of tiff files
        crop : 1 to crop image using cropping parameters, else 0
        register : 1 to register images
        img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2 : Number of pixels to crop on all sides
        medianfilter_window : window size for median filter
        start_frame, end_frame : Specify start and end frame to load just a subset of images


        """

        # Load data
        data = self.thundercontext.loadImages(FileName, inputFormat='tif', startIdx=start_frame, stopIdx=end_frame)
        data = data.medianFilter(size=medianfilter_window)  # Run a meidan filter through the images

        if crop:  # If crop is true, crop the data
            print 'Cropping'
            data = self.crop_data(data, img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2)
        if register:  # If register is true, register the data and plot
            print 'Registering Images'
            data = self.register_images(data, pdffile)

        if register == 1:
            self.saveasseries(data, 'registered_data')  # Convert data to series and save
        else:
            self.saveasseries(data, 'raw_data')

    def crop_data(self, data, img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2):

        """
        Crop Unwanted pixels in the data

        Parameters
        ----------
        data : Image dataset to crop
        img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2 : Number of pixels to crop on all sides

        Returns
        -------
        cropped data : Cropped images

        """
        # Cropping unwanted pixels if required
        img_shape = data.first()[1].shape
        print img_shape
        # If data has many planes
        if self.multiplane:
            cropped_data = data.crop((img_size_crop_x1, img_size_crop_y1, 0),
                                     (img_shape[0] - img_size_crop_x2, img_shape[1] - img_size_crop_y2, img_shape[2]))
        else:
            cropped_data = data.crop((img_size_crop_x1, img_size_crop_y1),
                                     (img_shape[0] - img_size_crop_x2, img_shape[1] - img_size_crop_y2))
        return cropped_data

    def background_subtraction(self, data):
        """
        Subtract background. Use the left hand corner to get the background part of the image

        Parameters
        ----------
        data : Series data set for background subtration

        Returns
        -------
        subtracted_data : Background subtracted image
        bg_trace : Average time series trace from the specified background

        """

        bg_trace = data.meanByRegions(
            [[(0, 20), (0, 50)]])  # This uses the top left hand corner as a background. Change as necessary.
        subtracted_data = data.toTimeSeries().bg_subtract(bg_trace.first()[1])
        self.saveasseries(subtracted_data, 'bgubtracted_data')

        return subtracted_data, bg_trace

    def detrend_data(self, data, detrend_order=10):
        """
        Detrend data using a non linear polynomial of order specified

        Parameters
        ----------
        data : Series dataset to detrend

        Returns
        -------
        detrended_data : Series data set after detrend

        """
        detrended_data = data.toTimeSeries().detrend(method='nonlin', order=detrend_order)
        self.saveasseries(data, 'detrended_data')

        return detrended_data

    def register_images(self, data, pdffile):
        """
        Register Image using cross correlation

        Parameters
        ----------
        data : Image dataset for registration
        pdffile : image file to create a plot to show unregistered and registered image and their difference.

        Returns
        -------
        corrected : Image dataset after registration

        """
        reg = Registration('crosscorr')
        reg.prepare(data, startIdx=0,
                    stopIdx=200)  # Take first 200 images to make template to register against. Change according to data

        model = reg.fit(data)
        corrected = model.transform(data)

        # Plot original and registered image
        original_mean = data.mean()
        corrected_mean = corrected.mean()

        fig1 = plt.figure(figsize=(10, 15))
        if self.multiplane:
            for ii in xrange(0, 4):
                self.plot_registered_images(fig1, ii, original_mean, corrected_mean, pdffile)
        else:
            ind = 1
            self.plot_registered_images(fig1, ind, original_mean, corrected_mean, pdffile)

        return corrected

    @staticmethod
    def plot_registered_images(fig1, ind, original, corrected, pdffile):
        # Plot the means and differences for viewing
        plt.subplot(ind, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.axis('off')
        plt.title('Original Image')
        plt.subplot(ind, 3, 2)
        plt.imshow(corrected, cmap='gray')
        plt.axis('off')
        plt.title('Registered Image')
        plt.subplot(ind, 3, 3)
        plt.imshow(corrected - original, cmap='gray')
        plt.axis('off')
        plt.title('Difference')
        plt.show()
        pdffile.savefig(fig1, bbox_inches='tight')

    def plotimageplanes(self, fig1, gs, img, plot_title='Habenula', gridspecs='[0,0]'):
        """
        Function to plot images in grayscale according to specified grid for easy formatting

        Parameters
        ----------
        fig1 : Figure handle
        gs : Grid Handle
        img : Image [2d or 3d image]
        plot_title : Title of the plot
        gridspecs : Grid to plot in

        """

        # If image has more than one plane, calculate number of subplots and plot
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(plot_title, fontsize=14)

    def normalize(self, data, squelch_parameter=0):
        """
        Squelch to remove noisy pixels, normalize using user standard deviation

        Parameters
        ----------
        data : Series dataset
        squelch_parameter : threshold below which all pixels will be removed

        Returns
        -------
        zscore_data : data after normalizing
        """
        zscore_data = data.squelch(threshold=squelch_parameter).center(axis=0).zscore(axis=0)
        # zscore_data.cache() #Cache if you have enough memory
        # zscore_data.dims
        self.saveasseries(zscore_data, 'zscore_data')
        return zscore_data

    def standardize(self, data, squelch_parameter=0, perc=10):
        """
        Squelch to remove noisy pixels, standardize using mean

        Parameters
        ----------
        data : Series dataset
        squelch_parameter : threshold below which all pixels will be removed

        Returns
        -------
        norm_data : data after normalizing
        """
        norm_data = data.center(axis=1).toTimeSeries().normalize(baseline='mean', perc=perc)
        self.saveasseries(norm_data, 'normalized_data')
        return norm_data

    def loadseriesdataset(self, savefilename):
        # Load series object given filename
        if os.path.exists(self.saveseriesdirectory + savefilename):
            print 'Loading pre saved series dataset'
            data = self.thundercontext.loadSeries(self.saveseriesdirectory + savefilename)
            return data
        else:
            raise ValueError('No such series object exists')

    def saveasseries(self, data, savefilename):
        # Save data as series object, given filename
        print 'Saving as series dataset from' + savefilename
        data.saveAsBinarySeries(self.saveseriesdirectory + savefilename, overwrite=True)

    @staticmethod
    def get_small_subset_for_plotting(data, number_samples=100, threshold=3):
        """
        Get a subset for plotting

        Parameters
        ----------
        data : Series dataset
        number_samples : Number of samples to get
        threshold : Threshold above which traces will be obtained

        Returns
        -------
        examples : subset of data
        """

        examples = data.subset(nsamples=number_samples, thresh=threshold)
        return examples

    def plot_traces(self, fig1, gs, plotting_data, gridspecs='[0,0]', **kwargs):
        """
        Function to plot traces according to specified grid for easy formatting

        Parameters
        ----------
        fig1 : Figure handle
        gs : Grid Handle
        plotting data : 2d array of traces to plot
        gridspecs : Grid to plot in

        **kwargs
        plot_title : Title of the plot

        """
        # Data : rows are cells, column is time
        with sns.axes_style('darkgrid'):
            ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
            plt.plot(plotting_data.T)
            plt.plot(mean(plotting_data, 0), 'k', linewidth=2)
            if 'plot_title' in kwargs:
                plt.title(kwargs.values()[0], fontsize=14)
            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            self.plot_stimulus_patch(ax1)

    def plot_vertical_lines_onset(self):
        # Draw a vertical line at stimulus onset
        for ii in xrange(0, size(self.stimulus_on_time)):
            plt.axvline(x=self.stimulus_on_time[ii], linestyle='-', color='k', linewidth=2)

    def plot_vertical_lines_offset(self):
        # Draw a vertical line at sitmulus offset
        for ii in xrange(0, size(self.stimulus_off_time)):
            plt.axvline(x=self.stimulus_off_time[ii], linestyle='--', color='k', linewidth=2)

    def plot_stimulus_patch(self, ax1):
        # Draw a patch according to light stimulus being delivered

        # Get y grid size
        y_tick = ax1.get_yticks()
        y_tick_width = y_tick[2] - y_tick[1]

        # Adjust axis to draw patch
        y_lim = ax1.get_ylim()
        ax1.set_ylim((y_lim[0], y_lim[1] + y_tick_width))

        for ii in xrange(0, size(self.stimulus_on_time)):
            # Find time of stimulus for width of patch
            time_stim = self.stimulus_off_time[ii] - self.stimulus_on_time[ii]

            # Check different cases of stimuli to create patches
            if self.stimulus_train[ii] == "Blue":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2, fc='b')
            elif self.stimulus_train[ii] == "StrongerBlue":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2,
                                          fc='darkblue')
            elif self.stimulus_train[ii] == "StrongerRed":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2,
                                          fc='darkred')
            elif self.stimulus_train[ii] == "Red":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2, fc='red')
            elif self.stimulus_train[ii] == "UV":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2, fc='aqua')
            elif self.stimulus_train[ii] == "US":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2,
                                          fc='orange')
            # Add Patch
            ax1.add_patch(rectangle)
