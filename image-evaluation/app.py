from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# تابع برای محاسبه MSE
def calculate_mse(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    return mse

# تابع برای محاسبه PSNR
def calculate_psnr(original, predicted):
    mse = calculate_mse(original, predicted)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# تابع برای محاسبه SSIM
def calculate_ssim(original, predicted):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    predicted_gray = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(original_gray, predicted_gray, full=True)
    return score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'real_image' not in request.files or 'predicted_image' not in request.files:
        return redirect(request.url)

    real_image = request.files['real_image']
    predicted_image = request.files['predicted_image']

    if real_image.filename == '' or predicted_image.filename == '':
        return redirect(request.url)

    real_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'real_image.jpg')
    predicted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_image.jpg')

    real_image.save(real_image_path)
    predicted_image.save(predicted_image_path)

    # بارگذاری تصاویر برای ارزیابی
    real_image = cv2.imread(real_image_path)
    predicted_image = cv2.imread(predicted_image_path)

    # تغییر اندازه تصاویر برای تطبیق
    real_image = cv2.resize(real_image, (predicted_image.shape[1], predicted_image.shape[0]))

    # محاسبه معیارهای ارزیابی
    mse_value = calculate_mse(real_image, predicted_image)
    psnr_value = calculate_psnr(real_image, predicted_image)
    ssim_value = calculate_ssim(real_image, predicted_image)

    return render_template('results.html', mse=mse_value, psnr=psnr_value, ssim=ssim_value)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)