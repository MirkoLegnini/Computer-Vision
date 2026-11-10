# Fruit Inspection

Project work for the Image Processing and Computer Vision course, Master's Degree, University of Bologna.

The project implements a classical image processing pipeline for the inspection of fruit. Starting from
near-infrared (NIR) and color images, the fruit is segmented from the background and surface defects are
located. No machine learning is used: every step relies on filtering, thresholding, morphology and
contour analysis with OpenCV.

<p align="center">
  <img src="first%20task/C0_000001.png" width="300">
  <img src="first%20task/C1_000001.png" width="300">
</p>

## Task 1: Fruit segmentation and defect detection

The goal is to outline the defects on NIR images of apples. The task is split into two parts.

The first part produces a binary mask of the apple (function `create_binary_mask`):

1. Gaussian filter to reduce noise
2. Exponential and linear stretching to increase contrast
3. Otsu thresholding to select the threshold automatically
4. Median filter to remove small residual noise
5. Blob detection followed by flood fill to close holes inside the mask

The second part detects the defects. The masked image is blurred and stretched, then Canny's edge detector
is applied. A morphological closing with an elliptical kernel closes small gaps in the edges. Finally only
closed contours are kept and they are filtered by area, so that the outer contour of the fruit is discarded
and only the contours around the defects remain.

## Task 2: Russet detection

The goal is to identify russet regions on the color images of apples.

Simple thresholding produced many false positives, so russet pixels are identified through the Mahalanobis
distance. Mean and covariance estimated on the two test images alone gave poor results, so they are computed
on a small set of russet samples cropped from the original images (folder `second task/stats`, function
`create_stats`).

Images are converted to the LAB color space and the L channel is discarded, which makes the result less
dependent on lighting. The background is removed with a binary mask, the distance map is normalized and an
exponential operator is applied to improve the histogram before Otsu thresholding.

The result is not fully accurate, but it identifies the russet regions reasonably well. Further processing or
parameter tuning could improve it.

## Final challenge: Kiwi inspection

The same approach is applied to five kiwi images (folder `final challenge`).

The mask is computed as in Task 1 (function `create_binary_mask_2`), with one additional step: since the
processing also produced some spots outside the fruit, blob detection is used on the mask to remove every
white region except the largest one. Defects are then detected with the same edge detection and contour
filtering used in Task 1.

## Repository structure

```
Computer-Vision/
├── Computer_Vision_Project.ipynb   final report with code, explanations and results
├── Computer_Vision_Project.py      script version used during development
├── first task/                     Task 1 images (apples)
├── second task/                    Task 2 images (apples with russet)
│   └── stats/                      russet samples used for mean and covariance
└── final challenge/                final challenge images (kiwis)
```

Files starting with `C0` are NIR images, files starting with `C1` are the corresponding color images.

## Getting started

Requirements: Python 3.11, OpenCV, NumPy, Matplotlib and Jupyter.

```bash
git clone https://github.com/MirkoLegnini/Computer-Vision.git
cd Computer-Vision
pip install opencv-python numpy matplotlib notebook
```

The notebook is the main deliverable and shows every task with intermediate results:

```bash
jupyter notebook Computer_Vision_Project.ipynb
```

The script can also be run directly:

```bash
python Computer_Vision_Project.py
```

In the script, the flags `show_images`, `show_images_2` and `show_images_3` enable the intermediate plots for
Task 1, Task 2 and the final challenge. Both files load images through relative paths, so they must be run
from the repository root.

## Authors

- Antonio Rapallini
- Mirko Legnini
- Dante Piotto
