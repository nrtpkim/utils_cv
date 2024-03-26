import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import os

model = YOLO('./asset/cctv-feed-ml_v3-post-prd-022024_yolov8l.pt')

class_color = {
    0 : (244,244,244),
    1 : (186,136,74)
}

# ==========================================================

### Preprocess resize
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

### draw bdbox
def draw_bounding_boxes(image, ret):
    bounding_boxes = ret['bdbox']
    class_conf = ret['conf']
    class_labels = ret['class']
    

    for box, conf, cls_no in zip(bounding_boxes,class_conf,class_labels):
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), class_color[cls_no], 1)  # Draw rectangle
        cv2.putText(image, f"{cls_no}: {round(conf, 3)}", (x_min, y_min - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, class_color[cls_no], 1)  # Draw class label

# ==========================================================
### get video name from batch run
video_name_arr = os.listdir('./test/25-3-2024')
print(video_name_arr)


# ==========================================================
### run infernece video
for video_name in video_name_arr:


    out = cv2.VideoWriter(f'./output_pred/video/{video_name}', cv2.VideoWriter_fourcc(*'XVID'), 2, (640,640))


    video_capture = cv2.VideoCapture(f'./test/25-3-2024/{video_name}')

    # Check if the video opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open the video file.")
        exit()

    fps = video_capture.get(cv2.CAP_PROP_FPS)


    # Read and display the video frame by frame
    frames_processed = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()  # Read a single frame
        if not ret:  # If no frame is read, the video has ended
            break
        
        frames_processed += 1

        if frames_processed % fps == 0: # sampling eery 1 sec

            frame = resize_letterbox(img=frame,new_size={'width':640,'height':640}).astype(np.uint8)

            ### run model ---> Paste you code below <---- 
            results = model.predict(frame)

            ret = []
            for output in results:
                output = output.cpu()

                ret.append({
                                "class": output.boxes.cls.to(int).tolist(),
                                "conf": output.boxes.conf.tolist(),
                                "bdbox": output.boxes.xyxy.to(int).tolist()
                            })
                
            if ret != []:
                draw_bounding_boxes(frame, ret[0])


            # Display the frame
            cv2.imshow('Video', frame)
            out.write(frame)


        # Check for the 'q' key press to quit the video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()