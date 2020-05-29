# Importing libraries
import glob
import os

import matplotlib
import nibabel as nib
import numpy as np

# Setting Matplotlib backend to Qt5, in order to use GUI interactively
matplotlib.use('Qt5Agg')
from matplotlib import pylab
import matplotlib.pyplot as plt


class MRIViewer:
    def __init__(self, brain, views=("axial", "sagittal", "coronal"), sig=10, hist_equ=False):
        """
        This class is an 3D MRI viewer with their subplots. Use Mouse scroll Up-Down or keyboard arrows to navigate
        between slices in different views. To see the image processing (fft, gaussian filtering etc.) press `y`key in
        figure. Different sub-plots can be added.

        :param brain:    3D NumPy array. It contains brain MRI data
        :param slice:    2D part NumPy array to be selected from the MRI image for the start.
                         It is preferred to leave it default
        :param views:    Name of the 3 subplots for three axis view
        :param sig:      Sigmoid value of the
        :param hist_equ: Equalize histogram if its `True`
        """

        # Initialize slice for 3 axis
        self.slice = []
        self.view = views
        self.hist_equ = hist_equ
        self.sig = sig
        # Gets all 3 axis view with swapping axes
        self.brain = self._set_of_view(brain)

        # Creates main frame and their axes
        self.figure, self.axes, self.frame = self._init_fig()

        # Connects mouse and keyboard events to the GUI
        for ax in self.axes:
            self.scr = ax.figure.canvas.mpl_connect('scroll_event', self.scroll)
            self.kpr = ax.figure.canvas.mpl_connect('key_press_event', self.key)

    def _init_fig(self):
        """
        Creates main window frame to see 3 axis of MRI view

        :return: Main window figure, axes and frame list
        """

        # Create 3 subplots, 3 row 1 column
        fig, axes = plt.subplots(1, 3, figsize=(8, 4), dpi=200)
        # Creates empty list to hold drawn images in it. If any image data update needed on main GUI's subplots.
        # Use this list
        frame = []
        for i, ax in enumerate(axes):
            # Taking init image slice
            if self.hist_equ:
                im = self.histeq(self.brain[i][:, :, self.slice[i]])[0]
            else:
                im = self.brain[i][:, :, self.slice[i]]
            ax.set_title(f"{self.view[i]}, Slice: {self.slice[i]}")
            # Plot and save it in list
            frame.append(ax.imshow(im, cmap="gray", origin="lower"))
        fig = pylab.gcf()
        # Setting main GUI title
        fig.canvas.set_window_title(f"MRI Views")
        fig.tight_layout()
        return fig, axes, frame

    def _set_of_view(self, brain):
        """
        MRI data is 3D numpy array.
        [x, y, z] -> Axial view
        [y, z, x] -> Sagittal view
        [x, z, y] -> Coronal view
        To get this arrays. We need to swap axes

        :param brain: 3D NumPy array
        :return:      List that contains all 3 axis views.
                    Need to navigate in last column to see difference between list items.
        """

        sagittal = np.array(np.swapaxes(brain, 0, -1))
        coronal = np.array(np.swapaxes(brain, 1, -1))
        axial = brain
        self.slice = [axial.shape[-1] // 2, sagittal.shape[-1] // 2, coronal.shape[-1] // 2]
        return [axial, sagittal, coronal]

    @staticmethod
    def create_gaussian_filters(size, sigma):
        """
        Creating gaussian filter desired size and with sigma

        :param size:  2D shape
        :param sigma: Integer number, can be use list to get multiple filters in one use
        :return: List of gaussian filters.
        """

        # Saves filters in this list
        fil = []
        for s in sigma:
            m, n = [(ss - 1.) / 2. for ss in size]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * s * s))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            fil.append(h)
        return fil

    @staticmethod
    def fft_shift(array):
        """
        Applies FFT-Shift to 2D array.

        :param array: 2D Array
        :return: FFT-Shifted 2D array
        """

        fil_fft = np.fft.fft2(array)
        return np.fft.fftshift(fil_fft)

    @staticmethod
    def apply_gaussian_iftt(fft_shft, filters):
        """
        Applies gaussian filters to fft shifted images. Then applies reverse fft on blurred image

        :param fft_shft: FFT-Shifted 2D array.
        :param filters: List of gaussian filters.
        :return: Blurred 2D Image
        """
        blurred = []
        for f in filters:
            blurred.append(np.fft.ifft2(fft_shft * f))
        return blurred

    @staticmethod
    def apply_high_boost(raw, gaussian_applied, k):
        """
        Only applies high boost in to guassian filtered 2D array

        :param raw: Raw 2D array
        :param gaussian_applied: Gaussian filter applied image
        :param k: Boost coefficient
        :return: Edge enhanced image
        """

        return ((k + 1) * raw) - (k * gaussian_applied)

    @staticmethod
    def histeq(image, n_bins=256):
        """
        This function equalize histogram of given 2D array

        :param image: 2D array
        :param n_bins: Number of bins to use in histogram equalization.
        :return: Equalized 2D array
        """

        im_hist, bins = np.histogram(image.flatten(), n_bins, density=True)
        cdf = im_hist.cumsum()
        cdf = n_bins * cdf / cdf[-1]
        img = np.interp(image.flatten(), bins[:-1], cdf)
        return img.reshape(image.shape), cdf

    def scroll(self, event):
        """
        Defines scroll events
        :param event:
        :return: Matplotlib event
        """
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                if event.button == "up" and self.slice[i] < self.brain[i].shape[-1] - 1:
                    self.__go_up(i)
                    break
                elif event.button == "down" and self.slice[i] > 0:
                    self.__go_down(i)
                    break

    def key(self, event):
        """
        Defines keyboard key events
        :param event:
        :return: Matplotlib event
        """
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                if event.key == "up" and self.slice[i] < self.brain[i].shape[-1] - 1:
                    self.__go_up(i)
                    break
                elif event.key == "down" and self.slice[i] > 0:
                    self.__go_down(i)
                    break
                elif event.key == "y":
                    self.__draw_subplots(i, self.sig)
                    break

    def __go_up(self, ax):
        self.slice[ax] += 1
        self.__set_data(ax)

    def __go_down(self, ax):
        self.slice[ax] -= 1
        self.__set_data(ax)

    def __set_data(self, ax):
        im = self.brain[ax][:, :, self.slice[ax]]
        if self.hist_equ:
            im = self.histeq(im)[0]
        self.frame[ax].set_data(im)
        self.axes[ax].set_title(f"{self.view[ax]}, Slice: {self.slice[ax]}")
        plt.draw()

    def __draw_subplots(self, ax, sigma):
        """
        Draws subplots if `y` key pressed in figure.
        :param ax: Selected figure axe
        :param sigma: Sigma value to used in gaussian filter
        """
        self.sub_fig, self.sub_axes = plt.subplots(2, 2, figsize=[7, 7])

        raw = self.brain[ax][:, :, self.slice[ax]]

        f_shift = self.fft_shift(raw)

        # Applies filters
        filt_ga = self.create_gaussian_filters(f_shift.shape[: 2], [sigma])
        out_gaussian = self.apply_gaussian_iftt(f_shift, filt_ga)
        k = 20
        out_high_boost = self.apply_high_boost(raw, out_gaussian[0], k=k)
        # Plots figures
        self.sub_axes[0, 0].imshow(raw, cmap="gray", origin="lower")
        self.sub_axes[0, 0].set_title("Raw Image")
        self.sub_axes[0, 1].imshow(50 * np.log(abs(1 + f_shift)), cmap="gray", origin="lower")
        self.sub_axes[0, 1].set_title("2D-FFT Shift")
        self.sub_axes[1, 0].imshow(abs(out_high_boost), cmap="gray", origin="lower")
        self.sub_axes[1, 0].set_title(f"High Boost - Sigma: {sigma} - k: {k}")
        self.sub_axes[1, 1].imshow(abs(out_gaussian[0]), cmap="gray", origin="lower")
        self.sub_axes[1, 1].set_title(f"Gaussian - Sigma: {sigma}")

        self.sub_fig.tight_layout()
        # Setting usbplot titles
        title = f"{self.view[ax]}, Slice: {self.slice[ax]} | Subplots"
        self.sub_fig.suptitle(title)
        self.sub_fig.canvas.set_window_title(title)
        plt.show()


# Project data directory
dir = "data"
f_names = glob.glob(os.path.join(dir, "*"))
for ind, f in enumerate(f_names):
    print(f"[{ind}]: {f}")
# Selecting file
inp = input("Select file no:")
mri_img = nib.load(f_names[int(inp)])
mri_img_data = mri_img.get_fdata()

# Creating viewer instance
v = MRIViewer(mri_img_data, hist_equ=True, sig=20)
# Showing Figures
plt.show()
