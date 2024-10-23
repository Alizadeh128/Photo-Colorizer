# Image Colorization and Evaluation System
##### This project demonstrates an image colorization system that converts black and white images into color images using deep learning techniques. Additionally, it provides an interface to evaluate the quality of the colorized images compared to real color images by calculating metrics such as MSE (Mean Squared Error), PSNR (Peak Signal-to-Noise Ratio), and SSIM (Structural Similarity Index). The system is implemented using Flask, OpenCV, and Jinja2 for the web interface.

## Features
- Upload Interface: Allows users to upload real and colorized images.
- Image Processing: Uses OpenCV to process images.
- Model Evaluation: Calculates image similarity metrics (MSE, PSNR, and SSIM) to evaluate the quality of the colorized image.
- Responsive Web Interface: A simple and user-friendly web interface developed with Flask and HTML.

## How to Use
1. Upload Images:
- The real color image.
- The colorized image (generated by the deep learning model).
1. View Results:
- After uploading, click the "Submit" button.
- The evaluation results (MSE, PSNR, and SSIM) will be displayed on the next page.
### Evaluation Metrics
- MSE (Mean Squared Error): Measures the average squared difference between the original and predicted images.
- PSNR (Peak Signal-to-Noise Ratio): Measures the ratio between the maximum possible signal and the noise affecting the fidelity of the image.
- SSIM (Structural Similarity Index): Measures the similarity between two images based on luminance, contrast, and structure.
## Requirements
- Python 3.6+
- Flask
- OpenCV
- Scikit-Image
