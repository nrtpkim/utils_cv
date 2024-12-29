import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import logging as logger


def find_key_by_value(dictionary, value):
    try:
        return next(key for key, val in dictionary.items() if val == value)
    except Exception as e:
        logger.error(f"find_key_by_value : {e}.")


def yolo_to_xywh(box, img_width, img_height):
    try:
        center_x, center_y, box_width, box_height = box[0], box[1], box[2], box[3]
        xmin = max(0, (center_x - (box_width / 2)) * img_width)
        ymin = max(0, (center_y - (box_height / 2)) * img_height)
        xmax = min(img_width, (center_x + (box_width / 2)) * img_width)
        ymax = min(img_height, (center_y + (box_height / 2)) * img_height)
        return (int(xmin), int(ymin), int(xmax), int(ymax))
    except Exception as e:
        logger.error(f"yolo_to_xywh : {e}.")


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
        logger.error(f"yolo_to_cvcontours : {e}.")


def generate_class_colors(class_dict):
    try:
        main_colors = [
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]  # pre-difined colors for first 5 classes
        class_id_colors = {}
        for idx, class_id in enumerate(class_dict.keys()):
            if idx < len(main_colors):
                class_id_colors[class_id] = main_colors[idx]
            else:
                # Generate random color for classes more than 5
                random_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                class_id_colors[class_id] = random_color
        return class_id_colors

    except Exception as e:
        logger.error(f"generate_class_colors : {e}.")


def get_min_class_counts_from_df(df):
    try:
        # Count the occurrences of each class
        class_count = df["label"].value_counts()
        class_count = dict(class_count.items())
        return min(class_count.values())

    except Exception as e:
        logger.error(f"get_min_class_counts_from_df : {e}.")


class ImageVisualizePlot:
    def __init__(
        self,
        image_label_df,
        class_dict,
        num_samples_per_class=5,
        show=True,
        save=True,
        save_dir="assets/results",
        mask_alpha=0.5,
    ):
        try:
            self.image_label_df = image_label_df
            self.class_dict = class_dict
            self.num_samples_per_class = np.min(
                [
                    num_samples_per_class,
                    int(get_min_class_counts_from_df(self.image_label_df)),
                ]
            )
            self.save = save
            self.save_dir = save_dir
            self.show = show
            self.box_thickness = 3
            self.save_visualize_result = "sample_images.png"
            self.mask_weight_show = mask_alpha
            logger.info("[utils-ImageVisualizePlot] Init Complete")

        except Exception as e:
            logger.error(f"[utils-ImageVisualizePlot] Init error : {e}.")

    # ploting boxes
    def __plot_bounding_boxes(
        self,
        image,
        label_path,
        class_dict,
        current_class_id,
        color_dict,
        ax,
        box_thickness,
    ):
        try:
            img_h, img_w, _ = image.shape

            if label_path is not None:
                with open(label_path, "r") as file:
                    lines = file.readlines()

                # Iterate each bbox label
                for line in lines:
                    bbox_info = line.strip().split()
                    bbox_yolo = [float(x) for x in bbox_info][1:]
                    class_id = int(bbox_info[0])
                    color = (
                        (0, 255, 0)
                        if class_id == current_class_id
                        else color_dict.get(class_id)
                    )  # current_class (green) vs others (random colors)
                    (x_min, y_min, x_max, y_max) = yolo_to_xywh(bbox_yolo, img_w, img_h)
                    cv2.rectangle(
                        image,
                        (x_min, y_min),
                        (x_max, y_max),
                        color,
                        thickness=box_thickness,
                    )
                    cv2.putText(
                        image,
                        class_dict[class_id],
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        color,
                        4,
                    )

            ax.imshow(image)
            ax.axis("off")

        except Exception as e:
            logger.error(f"[utils-ImageVisualizePlot] __plot_bounding_boxes : {e}.")

    def __plot_instance_masks(
        self,
        image,
        label_path,
        class_dict,
        current_class_id,
        color_dict,
        ax,
        box_thickness,
        mask_weight_show,
    ):
        try:
            img_h, img_w, _ = image.shape
            mask = np.zeros_like(image)

            if label_path is not None:
                with open(label_path, "r") as file:
                    lines = file.readlines()

                # Iterate each mask label
                for line in lines:
                    contour_info = line.strip().split()
                    contour_yolo = [float(x) for x in contour_info][1:]
                    contour_cv = yolo_to_cvcontours(contour_yolo, img_w, img_h)
                    class_id = int(contour_info[0])
                    color = (
                        (0, 255, 0)
                        if class_id == current_class_id
                        else color_dict.get(class_id)
                    )  # current_class (green) vs others (random colors)

                    # Draw contour on mask
                    cv2.drawContours(
                        mask, [contour_cv], -1, color, thickness=cv2.FILLED
                    )  # current_class (green) vs others (random colors)

                    # Draw bounding box around contour with class name
                    x, y, w, h = cv2.boundingRect(contour_cv)
                    cv2.rectangle(
                        image, (x, y), (x + w, y + h), color, thickness=box_thickness
                    )
                    cv2.putText(
                        image,
                        class_dict[class_id],
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        color,
                        4,
                    )

            # Plot the overlay image with labels
            ax.imshow(image)
            ax.imshow(mask, alpha=mask_weight_show)
            ax.axis("off")

        except Exception as e:
            logger.error(f"[utils-ImageVisualizePlot] __draw_instance_masks : {e}.")

    # CLASSIFICATION plot
    def display_images_with_labels(self):
        try:
            unique_classes = self.image_label_df["label"].unique()
            num_classes = len(unique_classes)
            _, axes = plt.subplots(
                num_classes, self.num_samples_per_class, figsize=(15, 10)
            )

            # Iterate through unique class
            for i, class_label in enumerate(unique_classes):
                class_df = self.image_label_df[
                    self.image_label_df["label"] == class_label
                ]
                sampled_indices = random.sample(
                    range(len(class_df)), self.num_samples_per_class
                )
                sampled_images = class_df.iloc[sampled_indices]["image_path"].values

                # plot each sample images
                for j, image_path in enumerate(sampled_images):
                    img = plt.imread(image_path)
                    axes[i, j].imshow(img)
                    axes[i, j].axis("off")
                    axes[i, j].set_title(f"{class_label}")
            plt.tight_layout()

            if self.save:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                plt.savefig(os.path.join(self.save_dir, self.save_visualize_result))
            if self.show:
                plt.show()

            logger.info(
                "[utils-ImageVisualizePlot] display_images_with_labels: Image Visualization Complete"
            )

        except Exception as e:
            logger.error(
                f"[utils-ImageVisualizePlot] display_images_with_labels : {e}."
            )

    # DETECTION plot
    def display_images_with_bboxes(self):
        try:
            unique_classes = self.image_label_df["label"].unique()
            num_classes = len(unique_classes)
            _, axes = plt.subplots(
                num_classes, self.num_samples_per_class, figsize=(15, 10)
            )

            # get color from class dict
            color_dict = generate_class_colors(self.class_dict)

            # Iterate through unique class
            for i, class_label in enumerate(unique_classes):
                current_class_id = find_key_by_value(
                    self.class_dict, class_label
                )  # get current class id
                class_df = self.image_label_df[
                    self.image_label_df["label"] == class_label
                ]
                sampled_indices = random.sample(
                    range(len(class_df)), self.num_samples_per_class
                )
                sampled_images = class_df.iloc[sampled_indices]["image_path"].values
                sampled_label_paths = class_df.iloc[sampled_indices][
                    "label_path"
                ].values

                # plot each sample images
                for j, (image_path, label_path) in enumerate(
                    zip(sampled_images, sampled_label_paths)
                ):
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    self.__plot_bounding_boxes(
                        image,
                        label_path,
                        self.class_dict,
                        current_class_id,
                        color_dict,
                        axes[i, j],
                        self.box_thickness,
                    )
                    axes[i, j].set_title(f"{class_label}")
            plt.tight_layout()

            if self.save:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                plt.savefig(os.path.join(self.save_dir, self.save_visualize_result))
            if self.show:
                plt.show()

            logger.info(
                "[utils-ImageVisualizePlot] display_images_with_bboxes: Image Visualization Complete"
            )

        except Exception as e:
            logger.error(
                f"[utils-ImageVisualizePlot] display_images_with_bboxes : {e}."
            )

    # INSTANCE SEGMENTATION plot
    def display_images_with_masks(self):
        try:
            unique_classes = self.image_label_df["label"].unique()
            num_classes = len(unique_classes)
            _, axes = plt.subplots(
                num_classes, self.num_samples_per_class, figsize=(15, 10)
            )

            # get color from class dict
            color_dict = generate_class_colors(self.class_dict)

            # Iterate through unique class
            for i, class_label in enumerate(unique_classes):
                current_class_id = find_key_by_value(self.class_dict, class_label)

                # randomly select images which label = current_class_id
                class_df = self.image_label_df[
                    self.image_label_df["label"] == class_label
                ]
                sampled_indices = random.sample(
                    range(len(class_df)), self.num_samples_per_class
                )
                sampled_images = class_df.iloc[sampled_indices]["image_path"].values
                sampled_label_paths = class_df.iloc[sampled_indices][
                    "label_path"
                ].values

                # Plot each sample images
                for j, (image_path, label_path) in enumerate(
                    zip(sampled_images, sampled_label_paths)
                ):
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    self.__plot_instance_masks(
                        image,
                        label_path,
                        self.class_dict,
                        current_class_id,
                        color_dict,
                        axes[i, j],
                        self.box_thickness,
                        self.mask_weight_show,
                    )
                    axes[i, j].set_title(f"{class_label}")

            plt.tight_layout()

            if self.save:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                plt.savefig(os.path.join(self.save_dir, self.save_visualize_result))
            if self.show:
                plt.show()

                logger.info(
                    "[utils-ImageVisualizePlot] display_images_with_masks: Image Visualization Complete"
                )

        except Exception as e:
            logger.error(f"[utils-ImageVisualizePlot] display_images_with_masks : {e}.")
