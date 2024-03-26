import numpy as np
import cv2
import matplotlib.pyplot as plt

def resize_letterbox(img: np.ndarray, new_size: dict, fill_value: int=114, channel: int = 3) -> np.ndarray:

    aspect_ratio = min(new_size['height'] / img.shape[1], new_size['width'] / img.shape[0])
    
    new_size_with_ar = int(img.shape[1] * aspect_ratio), int(img.shape[0] * aspect_ratio)
    
    resized_img = np.asarray(cv2.resize(img, new_size_with_ar, interpolation = cv2.INTER_AREA)) # by defaul interpolation = cv2.INTER_LINEAR // Yolov8 use cv2.INTER_AREA
    resized_h, resized_w, _ = resized_img.shape
    
    padded_img = np.full((new_size['height'], new_size['width'], channel), fill_value)

    center_x = new_size['width'] / 2
    center_y = new_size['height'] / 2
    
    x_range_start = int(center_x - (resized_w / 2))
    x_range_end = int(center_x + (resized_w / 2))
    
    y_range_start = int(center_y - (resized_h / 2))
    y_range_end = int(center_y + (resized_h / 2))
    
    padded_img[y_range_start: y_range_end, x_range_start: x_range_end, :] = resized_img
    return padded_img


if __name__ == "__main__":
    image = cv2.imread('./my_images.jpg')
    
    image = resize_letterbox(img=image, new_size={'width':640,'height':640}).astype(np.uint8)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()