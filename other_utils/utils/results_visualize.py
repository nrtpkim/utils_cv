import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import matplotlib.patches as mpatches
from ultralytics import YOLO
import logging as logger


def find_key_by_value(dictionary, value):
    try:
        return next(key for key, val in dictionary.items() if val == value)
    except Exception as e:
        logger.error(f"[utils-results_visualize] find_key_by_value : {e}.")


def yolo_to_xyxy(box, img_width, img_height):
    try:
        center_x, center_y, box_width, box_height = box[0], box[1], box[2], box[3]
        xmin = max(0, (center_x - (box_width / 2)) * img_width)
        ymin = max(0, (center_y - (box_height / 2)) * img_height)
        xmax = min(img_width, (center_x + (box_width / 2)) * img_width)
        ymax = min(img_height, (center_y + (box_height / 2)) * img_height)
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    except Exception as e:
        logger.error(f"[utils-results_visualize] yolo_to_xyxy : {e}.")


def plot_actual_bbox(image, label_path, current_class, class_dict):
    try:
        img_h, img_w, _ = image.shape
        current_class_id = find_key_by_value(class_dict, current_class)

        with open(label_path, "r") as file:
            lines = file.readlines()

        # Iterate each bbox label
        for line in lines:
            bbox_info = line.strip().split()
            class_id = int(bbox_info[0])
            if class_id == current_class_id:
                bbox_yolo = [float(x) for x in bbox_info][1:]
                color = (255, 0, 0)
                (x_min, y_min, x_max, y_max) = yolo_to_xyxy(bbox_yolo, img_w, img_h)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)
                # cv2.putText(image, current_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        return image

    except Exception as e:
        logger.error(f"[utils-results_visualize] plot_actual_bbox : {e}.")


def plot_predict_bbox(image, image_path, model, current_class, class_dict):
    try:
        img_h, img_w, _ = image.shape
        current_class_id = find_key_by_value(class_dict, current_class)

        results = model.predict(image_path, iou=0.6, conf=0.25, verbose=False)
        boxes = results[0].cpu().boxes
        if boxes != None:
            for num, box in enumerate(boxes):
                class_id = int(box.cls.numpy()[0])
                bbox_yolo = [
                    box.xywhn[0].numpy()[0],
                    box.xywhn[0].numpy()[1],
                    box.xywhn[0].numpy()[2],
                    box.xywhn[0].numpy()[3],
                ]
                if class_id == current_class_id:
                    color = (0, 255, 0)
                    (x_min, y_min, x_max, y_max) = yolo_to_xyxy(bbox_yolo, img_w, img_h)
                    cv2.rectangle(
                        image, (x_min, y_min), (x_max, y_max), color, thickness=3
                    )
                    # cv2.putText(image, current_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        return image

    except Exception as e:
        logger.error(f"[utils-results_visualize] plot_predict_bbox : {e}.")


def yolo_to_cvcontours(coordinates, img_width, img_height):
    try:
        # Ensure the number of coordinates is even
        assert len(coordinates) % 2 == 0

        contours = []
        for i in range(0, len(coordinates), 2):
            x = coordinates[i] * img_width
            y = coordinates[i + 1] * img_height
            contours.append(tuple([int(x), int(y)]))

        return np.array(contours)

    except Exception as e:
        logger.error(f"[utils-results_visualize] yolo_to_cvcontours : {e}.")


def plot_actual_mask(image, label_path, current_class, class_dict):
    try:
        img_h, img_w, _ = image.shape
        mask = np.zeros_like(image)
        current_class_id = find_key_by_value(class_dict, current_class)

        with open(label_path, "r") as file:
            lines = file.readlines()

        # Iterate each bbox label
        for line in lines:
            contour_info = line.strip().split()
            class_id = int(contour_info[0])
            contour_yolo = [float(x) for x in contour_info][1:]
            if class_id == current_class_id:
                contour_cv = yolo_to_cvcontours(contour_yolo, img_w, img_h)
                color = (255, 0, 0)
                cv2.drawContours(mask, [contour_cv], -1, color, thickness=cv2.FILLED)

                # Draw bounding box around contour with class name
                x, y, w, h = cv2.boundingRect(contour_cv)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)
                # cv2.putText(image, current_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        return image, mask

    except Exception as e:
        logger.error(f"[utils-results_visualize] plot_actual_mask : {e}.")


def plot_predict_mask(image, image_path, model, current_class, class_dict):
    try:
        img_h, img_w, _ = image.shape
        mask_img = np.zeros_like(image)
        current_class_id = find_key_by_value(class_dict, current_class)

        results = model.predict(image_path, iou=0.6, conf=0.25, verbose=False)
        masks = results[0].cpu().masks
        boxes = results[0].cpu().boxes
        if masks != None:
            for num, mask in enumerate(masks):
                class_id = int(boxes[num].cls.numpy()[0])
                contour_yolo = mask.xyn[0].flatten()
                if class_id == current_class_id:
                    contour_cv = yolo_to_cvcontours(contour_yolo, img_w, img_h)
                    color = (0, 255, 0)
                    cv2.drawContours(
                        mask_img, [contour_cv], -1, color, thickness=cv2.FILLED
                    )

                    # Draw bounding box around contour with class name
                    x, y, w, h = cv2.boundingRect(contour_cv)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)
                    # cv2.putText(image, current_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        return image, mask_img

    except Exception as e:
        logger.error(f"[utils-results_visualize] plot_predict_mask : {e}.")


def visualize_classification_results(result_path, images_per_case):
    try:
        # load csv results (each image)
        csv_file_path = os.path.join(result_path, "result_prediction.csv")
        df = pd.read_csv(csv_file_path)

        unique_classes = df["class_name"].unique()
        num_classes = len(unique_classes)
        num_cases = 3 * images_per_case  # TP, FP, FN * images_per_case
        fig, axes = plt.subplots(
            num_classes, num_cases, figsize=(num_cases * 3, num_classes * 3)
        )

        for i, class_label in enumerate(unique_classes):

            # Find TP, FP, FN cases for this class
            tp_df = df[(df["class_name"] == class_label) & (df["pred"] == class_label)]
            fp_df = df[(df["class_name"] != class_label) & (df["pred"] == class_label)]
            fn_df = df[(df["class_name"] == class_label) & (df["pred"] != class_label)]

            # shuffle row
            tp_df = tp_df.sample(frac=1).reset_index(drop=True)
            fp_df = fp_df.sample(frac=1).reset_index(drop=True)
            fn_df = fn_df.sample(frac=1).reset_index(drop=True)

            # Plot True Positives examples
            for j, (_, row) in enumerate(tp_df.head(images_per_case).iterrows()):
                img = plt.imread(row["path"])
                axes[i, j].imshow(img)
                axes[i, j].axis("off")
                axes[i, j].set_title(f"{row['class_name']} | {row['pred']} (TP)")

            # Plot False Positives examples
            for j, (_, row) in enumerate(fp_df.head(images_per_case).iterrows()):
                img = plt.imread(row["path"])
                axes[i, images_per_case + j].imshow(img)
                axes[i, images_per_case + j].axis("off")
                axes[i, images_per_case + j].set_title(
                    f"{row['class_name']} | {row['pred']} (FP)"
                )

            # Plot False Negatives examples
            for j, (_, row) in enumerate(fn_df.head(images_per_case).iterrows()):
                img = plt.imread(row["path"])
                axes[i, images_per_case * 2 + j].imshow(img)
                axes[i, images_per_case * 2 + j].axis("off")
                axes[i, images_per_case * 2 + j].set_title(
                    f"{row['class_name']} | {row['pred']} (FN)"
                )

            # Hide axis ticks for subplots with no image
            for j in range(num_cases):
                axes[i, j].axis("off")

        plt.tight_layout()
        plt.show()

        return fig

    except Exception as e:
        logger.error(
            f"[utils-results_visualize] visualize_classification_results : {e}."
        )


def visualize_detection_results(result_path, images_per_case, class_dict):
    try:
        # load csv results (each image)
        csv_file_path = os.path.join(result_path, "result_prediction.csv")
        df = pd.read_csv(csv_file_path)

        unique_classes = df["class_name"].dropna().unique()  # unique class except None
        num_classes = len(unique_classes)
        num_cases = 3 * images_per_case  # TP, FP, FN * images_per_case
        fig, axes = plt.subplots(
            num_classes, num_cases, figsize=(num_cases * 3, num_classes * 3)
        )

        # load model
        model_file_path = os.path.join(
            result_path.replace("val", "train"), "weights/best.pt"
        )
        model = YOLO(model_file_path)

        for i, class_label in enumerate(unique_classes):

            # Find TP, FP, FN cases for this class
            tp_df = df[(df["class_name"] == class_label) & (df["pred"] == class_label)]
            fp_df = df[(df["class_name"] != class_label) & (df["pred"] == class_label)]
            fn_df = df[(df["class_name"] == class_label) & (df["pred"] != class_label)]

            # shuffle row
            tp_df = tp_df.sample(frac=1).reset_index(drop=True)
            fp_df = fp_df.sample(frac=1).reset_index(drop=True)
            fn_df = fn_df.sample(frac=1).reset_index(drop=True)

            # Plot True Positives examples
            for j, (_, row) in enumerate(tp_df.head(images_per_case).iterrows()):
                img = cv2.imread(row["path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # plot prediction
                img = plot_predict_bbox(
                    img, row["path"], model, class_label, class_dict
                )

                # plot actual label
                label_path = (
                    os.path.splitext(row["path"].replace("images", "labels"))[0]
                    + ".txt"
                )
                img = plot_actual_bbox(img, label_path, class_label, class_dict)

                axes[i, j].imshow(img)
                axes[i, j].axis("off")
                axes[i, j].set_title(f"{row['class_name']} (TP)")

            # Plot False Positives examples
            for j, (_, row) in enumerate(fp_df.head(images_per_case).iterrows()):
                img = cv2.imread(row["path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # plot prediction
                img = plot_predict_bbox(
                    img, row["path"], model, class_label, class_dict
                )

                # plot actual label
                label_path = (
                    os.path.splitext(row["path"].replace("images", "labels"))[0]
                    + ".txt"
                )
                img = plot_actual_bbox(img, label_path, class_label, class_dict)

                axes[i, images_per_case + j].imshow(img)
                axes[i, images_per_case + j].axis("off")
                axes[i, images_per_case + j].set_title(f"{row['pred']} (FP)")

            # Plot False Negatives examples
            for j, (_, row) in enumerate(fn_df.head(images_per_case).iterrows()):
                img = cv2.imread(row["path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # plot prediction
                img = plot_predict_bbox(
                    img, row["path"], model, class_label, class_dict
                )

                # plot actual label
                label_path = (
                    os.path.splitext(row["path"].replace("images", "labels"))[0]
                    + ".txt"
                )
                img = plot_actual_bbox(img, label_path, class_label, class_dict)

                axes[i, images_per_case * 2 + j].imshow(img)
                axes[i, images_per_case * 2 + j].axis("off")
                axes[i, images_per_case * 2 + j].set_title(f"{row['class_name']} (FN)")

            # Hide axis ticks for subplots with no image
            for j in range(num_cases):
                axes[i, j].axis("off")

        red_patch = mpatches.Patch(color="red", label="Actual")
        green_patch = mpatches.Patch(color="green", label="Prediction")
        plt.legend(
            handles=[red_patch, green_patch],
            bbox_to_anchor=(1, 0),
            loc="lower right",
            bbox_transform=fig.transFigure,
        )
        plt.tight_layout()
        plt.show()

        return fig
    except Exception as e:
        logger.error(f"[utils-results_visualize] visualize_detection_results : {e}.")


def visualize_instance_segmentation_results(result_path, images_per_case, class_dict):
    try:
        # load csv results (each image)
        csv_file_path = os.path.join(result_path, "result_prediction.csv")
        df = pd.read_csv(csv_file_path)

        unique_classes = df["class_name"].dropna().unique()
        num_classes = len(unique_classes)
        num_cases = 3 * images_per_case  # TP, FP, FN * images_per_case
        fig, axes = plt.subplots(
            num_classes, num_cases, figsize=(num_cases * 5, num_classes * 5)
        )

        # load model
        model_file_path = os.path.join(
            result_path.replace("val", "train"), "weights/best.pt"
        )
        model = YOLO(model_file_path)

        for i, class_label in enumerate(unique_classes):

            # Find TP, FP, FN cases for this class
            tp_df = df[(df["class_name"] == class_label) & (df["pred"] == class_label)]
            fp_df = df[(df["class_name"] != class_label) & (df["pred"] == class_label)]
            fn_df = df[(df["class_name"] == class_label) & (df["pred"] != class_label)]

            tp_df = tp_df.sample(frac=1).reset_index(drop=True)
            fp_df = fp_df.sample(frac=1).reset_index(drop=True)
            fn_df = fn_df.sample(frac=1).reset_index(drop=True)

            # Plot True Positives examples
            for j, (_, row) in enumerate(tp_df.head(images_per_case).iterrows()):
                img = cv2.imread(row["path"])  # recheck image_path
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # plot prediction
                img, mask_pred = plot_predict_mask(
                    img, row["path"], model, class_label, class_dict
                )

                # plot actual label
                label_path = (
                    os.path.splitext(row["path"].replace("images", "labels"))[0]
                    + ".txt"
                )
                img, mask = plot_actual_mask(img, label_path, class_label, class_dict)

                axes[i, j].imshow(img)
                axes[i, j].imshow(mask_pred, alpha=0.4)
                axes[i, j].imshow(mask, alpha=0.4)
                axes[i, j].axis("off")
                axes[i, j].set_title(f"{row['class_name']} (TP)")

            # Plot False Positives examples
            for j, (_, row) in enumerate(fp_df.head(images_per_case).iterrows()):
                img = cv2.imread(row["path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # plot prediction
                img, mask_pred = plot_predict_mask(
                    img, row["path"], model, class_label, class_dict
                )

                # plot actual label
                label_path = (
                    os.path.splitext(row["path"].replace("images", "labels"))[0]
                    + ".txt"
                )  # recheck label_path
                img, mask = plot_actual_mask(img, label_path, class_label, class_dict)

                axes[i, images_per_case + j].imshow(img)
                axes[i, images_per_case + j].imshow(mask_pred, alpha=0.4)
                axes[i, images_per_case + j].imshow(mask, alpha=0.4)
                axes[i, images_per_case + j].axis("off")
                axes[i, images_per_case + j].set_title(f"{row['pred']} (FP)")

            # Plot False Negatives examples
            for j, (_, row) in enumerate(fn_df.head(images_per_case).iterrows()):
                img = cv2.imread(row["path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # plot prediction
                img, mask_pred = plot_predict_mask(
                    img, row["path"], model, class_label, class_dict
                )

                # plot actual label
                label_path = (
                    os.path.splitext(row["path"].replace("images", "labels"))[0]
                    + ".txt"
                )
                img, mask = plot_actual_mask(img, label_path, class_label, class_dict)

                axes[i, images_per_case * 2 + j].imshow(img)
                axes[i, images_per_case * 2 + j].imshow(mask_pred, alpha=0.4)
                axes[i, images_per_case * 2 + j].imshow(mask, alpha=0.4)
                axes[i, images_per_case * 2 + j].axis("off")
                axes[i, images_per_case * 2 + j].set_title(f"{row['class_name']} (FN)")

            # Hide axis ticks for subplots with no image
            for j in range(num_cases):
                axes[i, j].axis("off")

        red_patch = mpatches.Patch(color="red", label="False Negative")
        green_patch = mpatches.Patch(color="green", label="False Positive")
        yellow_patch = mpatches.Patch(color="yellow", label="True Positive")
        plt.legend(
            handles=[red_patch, green_patch, yellow_patch],
            bbox_to_anchor=(1, 0),
            loc="lower right",
            bbox_transform=fig.transFigure,
        )
        plt.tight_layout()
        plt.show()

        return fig

    except Exception as e:
        logger.error(
            f"[utils-results_visualize] visualize_instance_segmentation_results : {e}."
        )
