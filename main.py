import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Tạo thư mục lưu kết quả nếu chưa có
if not os.path.exists('output'):
    os.makedirs('output')


def negative_image(image):
    """
    Tạo ảnh âm tính bằng công thức:
    negative_image = 255 - original_image
    """
    return 255 - image


def increase_contrast(image, alpha=2.0, beta=0):
    """
    Tăng độ tương phản sử dụng công thức tuyến tính:
    new_image = alpha * old_image + beta
    Trong đó:
    - alpha là hệ số tăng cường độ tương phản (mặc định là 2.0)
    - beta là độ dịch (mặc định là 0, không thay đổi độ sáng)
    """
    contrast_image = alpha * image + beta  # Tăng cường độ tương phản
    # Giới hạn giá trị pixel từ 0 đến 255
    contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)
    return contrast_image


def log_transform(image):
    """
    Biến đổi log sử dụng công thức:
    new_image = c * log(1 + old_image)
    Trong đó:
    - c là hằng số tỉ lệ, c = 255 / log(1 + max(old_image))
    """
    c = 255 / np.log(1 + np.max(image))  # Tính hằng số c dựa trên giá trị lớn nhất của ảnh
    log_image = c * np.log(1 + image)  # Biến đổi log
    return np.array(log_image, dtype=np.uint8)  # Trả về ảnh sau khi biến đổi


def histogram_equalization(image):
    """
    Cân bằng Histogram bằng công thức thủ công:
    1. Tính histogram của ảnh gốc.
    2. Tính hàm phân phối tích lũy (CDF).
    3. Chuẩn hóa CDF và ánh xạ lại các pixel theo CDF.
    """
    # Tính histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    # Tính hàm phân phối tích lũy (CDF)
    cdf = hist.cumsum()
    # Chuẩn hóa CDF
    cdf_normalized = cdf * 255 / cdf[-1]
    # Ánh xạ các giá trị pixel dựa trên CDF đã chuẩn hóa
    equalized_image = cdf_normalized[image.flatten()].reshape(image.shape).astype(np.uint8)
    return equalized_image


def display_images(original, negative, contrast, log, histogram):
    """
    Hiển thị ảnh gốc và các ảnh đã xử lý
    """
    images = [original, negative, contrast, log, histogram]
    titles = ['Original', 'Negative', 'Increased Contrast', 'Log Transform', 'Histogram Equalization']

    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_images(output_dir, original, negative, contrast, log, histogram):
    """
    Lưu các ảnh đã xử lý vào thư mục đầu ra
    """
    Image.fromarray(original).save(os.path.join(output_dir, 'original.jpg'))
    Image.fromarray(negative).save(os.path.join(output_dir, 'negative.jpg'))
    Image.fromarray(contrast).save(os.path.join(output_dir, 'contrast.jpg'))
    Image.fromarray(log).save(os.path.join(output_dir, 'log_transform.jpg'))
    Image.fromarray(histogram).save(os.path.join(output_dir, 'histogram_equalization.jpg'))


def main():
    # Đọc ảnh X-quang dưới dạng ảnh xám
    image = Image.open('images/xray.jpg').convert('L')
    image = np.array(image)

    # 1. Tạo ảnh âm tính
    negative = negative_image(image)

    # 2. Tăng độ tương phản
    contrast = increase_contrast(image)

    # 3. Biến đổi log
    log = log_transform(image)

    # 4. Cân bằng histogram
    histogram = histogram_equalization(image)

    # Hiển thị các ảnh đã xử lý
    display_images(image, negative, contrast, log, histogram)

    # Lưu các ảnh đã xử lý
    save_images('output', image, negative, contrast, log, histogram)


if __name__ == "__main__":
    main()
