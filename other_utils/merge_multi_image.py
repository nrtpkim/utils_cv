import os
import cv2
import numpy as np
from tqdm import tqdm

### Condiition to used: each image must in each folder & file name is same
# Ex. 
# -  ground_truth/{image_name}
# -  1_result/{image_name}
# -  2_result/{image_name}

### Get data img name
file_name = []
MAIN_DATA_FOLDER = 'test/round 2'
for root, dirs, files in os.walk(MAIN_DATA_FOLDER):
    for file in files:
        
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".avif")):
            image_path = os.path.join(root, file)
            file_name.append(file)

print(file_name)

# =============================
### Run merge
for img_name in tqdm(file_name):
    # Load your images
    image1 = cv2.imread(f'./ground_truth/{img_name}')
    image2 = cv2.imread(f'./1_result/{img_name}')
    image3 = cv2.imread(f'./2_result/{img_name}')


    # Check if images are loaded successfully
    if image1 is None or image2 is None or image3 is None:
        print("Error: Unable to load images.")
    else:
        # Resize images if they have different dimensions
        height = max(image1.shape[0], image2.shape[0], image3.shape[0])
        width1 = int(image1.shape[1] * (height / image1.shape[0]))
        width2 = int(image2.shape[1] * (height / image2.shape[0]))
        width3 = int(image3.shape[1] * (height / image3.shape[0]))
        
        image1 = cv2.resize(image1, (width1, height))
        image2 = cv2.resize(image2, (width2, height))
        image3 = cv2.resize(image3, (width3, height))

        # Add text below each image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)
        text1 = "Ground_truth"
        text2 = "#1_post_prd"
        text3 = "#2_post_prd"
        text_size1 = cv2.getTextSize(text1, font, font_scale, font_thickness)[0]
        text_size2 = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]
        text_size3 = cv2.getTextSize(text3, font, font_scale, font_thickness)[0]
        text_x1 = (image1.shape[1] - text_size1[0]) // 2
        text_x2 = (image2.shape[1] - text_size2[0]) // 2
        text_x3 = (image3.shape[1] - text_size3[0]) // 2
        text_y = height + text_size1[1] - 40  # distance from image bottom

        cv2.putText(image1, text1, (text_x1, text_y), font, font_scale, text_color, font_thickness)
        cv2.putText(image2, text2, (text_x2, text_y), font, font_scale, text_color, font_thickness)
        cv2.putText(image3, text3, (text_x3, text_y), font, font_scale, text_color, font_thickness)

        # Horizontally stack images
        stacked_image = np.hstack((image1, image2, image3))

        # Display the stacked image
        # cv2.imshow('Horizontal Stack', stacked_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ### edit outpath
        cv2.imwrite(f"output_pred/stack/{img_name}", stacked_image)
