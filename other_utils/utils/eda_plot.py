import os
import matplotlib.pyplot as plt
import pandas as pd
import logging as logger
import yaml
from utils.images_visualize_plot import ImageVisualizePlot


def get_class_name_from_yaml(yaml_path):

    try:
        # Load YAML file
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Extract names from YAML data
        class_dict = data.get("names", {})

    except Exception as e:
        logger.error(f"[utils-eda_plot] get_class_name_from_yaml : {e}.")

    return class_dict


def plot_class_distribution(classes, counts, save_dir, show=True, save=True):

    try:

        plt.bar(classes, counts, align="center", alpha=0.7)
        plt.xlabel("Class name")
        plt.ylabel("Count")
        plt.title("Class Distribution")

        for i in range(len(classes)):
            plt.text(i, counts[i] + 0.1, str(counts[i]), ha="center")

        plt.xticks(rotation=45, ha="right")

        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, "class_distribution.png"))
        if show:
            plt.show()

        logger.info("[utils-eda_plot] plot_class_distribution : Plotting Complete")

    except Exception as e:
        logger.error(f"[utils-eda_plot] plot_class_distribution : {e}.")


def get_class_counts_from_df(df):

    try:
        classes = []
        counts = []

        # Count the occurrences of each class
        class_count = df["label"].value_counts()
        class_count = dict(class_count.items())

        # get class name and number of each class
        classes = list(class_count.keys())
        counts = list(class_count.values())

        logger.info("[utils-eda_plot] get_class_counts_from_df : get df Complete")

    except Exception as e:
        logger.error(f"[utils-eda_plot] get_class_counts_from_df : {e}.")

    return classes, counts


def get_df_from_yolo_label(main_data_folder, class_dict):

    try:
        # class dict for no object
        no_object_class = 999

        # image & label folder
        image_folder_list = [main_data_folder + "/images"]
        label_folders_list = [main_data_folder + "/labels"]

        image_paths = []
        image_classes = []
        label_paths = []

        # Iterate through all folders in the folder list
        for folder_idx, yolo_image_folder in enumerate(image_folder_list):
            # Iterate through all files in the folder
            for file_name in os.listdir(yolo_image_folder):
                if file_name.lower().endswith(
                    (".jpg", ".png", ".bmp", ".jpeg", ".webp", ".avif", ".gif")
                ):
                    image_path = os.path.join(yolo_image_folder, file_name)
                    label_path = os.path.join(
                        label_folders_list[folder_idx],
                        os.path.splitext(file_name)[0] + ".txt",
                    )

                    # Check if the .txt label file exists
                    if os.path.isfile(label_path):

                        with open(label_path, "r") as file:
                            lines = file.readlines()

                        for line in lines:
                            class_id = int(line.split()[0])
                            image_class = class_dict.get(class_id, f"Class_{class_id}")

                            # append data
                            image_paths.append(image_path)
                            image_classes.append(image_class)
                            label_paths.append(label_path)

                    # no object image
                    else:
                        if no_object_class not in class_dict:
                            class_dict[no_object_class] = (
                                "No object"  # class_id=999 for no object
                            )
                        image_class = "No object"
                        label_path = None

                        # append data
                        image_paths.append(image_path)
                        image_classes.append(image_class)
                        label_paths.append(label_path)

        # Create DataFrame containing image paths, classes, and label paths
        df = pd.DataFrame(
            {
                "image_path": image_paths,
                "label_path": label_paths,
                "label": image_classes,
            }
        )

        logger.info("[utils-eda_plot] get_df_from_yolo_label : get df Complete")

    except Exception as e:
        logger.error(f"[utils-eda_plot] get_df_from_yolo_label : {e}.")

    return df, class_dict


def get_df_from_csv(csv_path):
    try:

        df = pd.read_csv(csv_path)

        # Find the column name among 'label', 'class', or 'class_name'
        column_label = next(
            (col for col in ["label", "class", "class_name"] if col in df.columns), None
        )

        if column_label is None:
            raise ValueError(
                "None of the expected column names ('label', 'class', 'class_name') found in the DataFrame."
            )

        # Create DataFrame containing image paths and classes
        folder_path = os.path.dirname(csv_path)
        df["image_path"] = df["filename"].apply(
            lambda x: os.path.join(folder_path, os.path.basename(x))
        )
        df = pd.DataFrame({"image_path": df["image_path"], "label": df[column_label]})

        # Extract unique class labels
        unique_classes = sorted(df["label"].unique())

        # Map unique class labels to numerical indices
        class_dict = {idx: label for idx, label in enumerate(unique_classes)}

        logger.info("[utils-eda_plot] get_df_from_csv : get df Complete")

    except Exception as e:
        logger.error(f"[utils-eda_plot] get_df_from_csv : {e}.")

    return df, class_dict


def get_df_from_directory(folder_path):
    try:
        class_count = {}
        image_paths = []
        image_classes = []

        # Iterate through subfolders (each subfolder represents a class)
        for class_folder in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_folder)

            if os.path.isdir(class_path):
                # Count the number of images in each class
                class_count[class_folder] = len(os.listdir(class_path))

                # Add image paths and their classes to lists
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    image_paths.append(image_path)
                    image_classes.append(class_folder)

        # Create DataFrame containing image paths and labels
        df = pd.DataFrame({"image_path": image_paths, "label": image_classes})

        # Create class_dict
        class_dict = {idx: label for idx, label in enumerate(class_count.keys())}

        logger.info("[utils-eda_plot] get_df_from_directory : get df Complete")

    except Exception as e:
        logger.error(f"[utils-eda_plot] get_df_from_directory : {e}.")

    return df, class_dict


def classification_eda(
    input,
    save_path,
    class_distribution_plot=True,
    sample_images_visualize=True,
    visualize_per_class=5,
):
    try:
        if os.path.isdir(input):
            df, class_dict = get_df_from_directory(input)
        elif input.lower().endswith(".csv"):
            df, class_dict = get_df_from_csv(input)
        else:
            print(
                f"Input format in [{input}] is not supported for EDA in classification task"
            )
            return

        classes, counts = get_class_counts_from_df(df)

        if class_distribution_plot:
            plot_class_distribution(classes, counts, save_path, show=True, save=True)

        if sample_images_visualize:
            image_visualize = ImageVisualizePlot(
                df,
                class_dict,
                num_samples_per_class=visualize_per_class,
                save_dir=save_path,
            )
            image_visualize.display_images_with_labels()

        logger.info(
            "[utils-eda_plot] classification_eda: EDA for CLASSIFICATION TASK Complete"
        )

    except Exception as e:
        logger.error(f"[utils-eda_plot] classification_eda: {e}.")


def detection_eda(
    input_folder,
    class_dict,
    save_path,
    class_distribution_plot=True,
    sample_images_visualize=True,
    visualize_per_class=5,
):
    try:
        df, class_dict = get_df_from_yolo_label(input_folder, class_dict)
        classes, counts = get_class_counts_from_df(df)

        if class_distribution_plot:
            plot_class_distribution(classes, counts, save_path, show=True, save=True)

        if sample_images_visualize:
            image_visualize = ImageVisualizePlot(
                df,
                class_dict,
                num_samples_per_class=visualize_per_class,
                save_dir=save_path,
            )
            image_visualize.display_images_with_bboxes()

        logger.info("[utils-eda_plot] detection_eda: EDA for DETECTION TASK Complete")

    except Exception as e:
        logger.error(f"[utils-eda_plot] detection_eda: {e}.")


def instance_segmentation_eda(
    input_folder,
    class_dict,
    save_path,
    class_distribution_plot=True,
    sample_images_visualize=True,
    visualize_per_class=5,
):
    try:
        df, class_dict = get_df_from_yolo_label(input_folder, class_dict)
        classes, counts = get_class_counts_from_df(df)

        if class_distribution_plot:
            plot_class_distribution(classes, counts, save_path, show=True, save=True)

        if sample_images_visualize:
            image_visualize = ImageVisualizePlot(
                df,
                class_dict,
                num_samples_per_class=visualize_per_class,
                save_dir=save_path,
            )
            image_visualize.display_images_with_masks()

        logger.info(
            "[utils-eda_plot] instance_segmentation_eda: EDA for INSTANCE_SEGMENTATION TASK Complete"
        )

    except Exception as e:
        logger.error(f"[utils-eda_plot] instance_segmentation_eda: {e}.")


def eda_plot(
    input_folder,
    class_dict,
    task,
    save_path,
    class_distribution_plot=True,
    sample_images_visualize=True,
    visualize_per_class=5,
):
    try:
        if task == "CLASSIFICATION":
            classification_eda(
                input_folder,
                save_path,
                class_distribution_plot,
                sample_images_visualize,
                visualize_per_class,
            )
        elif task == "DETECTION":
            detection_eda(
                input_folder,
                class_dict,
                save_path,
                class_distribution_plot,
                sample_images_visualize,
                visualize_per_class,
            )
        elif task == "INSTANCE_SEGMENTATION":
            instance_segmentation_eda(
                input_folder,
                class_dict,
                save_path,
                class_distribution_plot,
                sample_images_visualize,
                visualize_per_class,
            )
        else:
            print(f"Task[{task}] not supported for EDA")
            logger.error(
                f"[utils-eda_plot] eda_plot: Task[{task}] not supported for EDA."
            )

    except Exception as e:
        logger.error(f"[utils-eda_plot] eda_plot: {e}.")
