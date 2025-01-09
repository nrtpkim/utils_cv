import yaml
import logging
from itertools import combinations
import os
import re
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def check_and_create_file(file_path):
    try:
        if not os.path.exists(file_path):
            with open(file_path, "w"):
                pass
            print(f"File '{file_path}' created successfully.")

    except Exception as e:
        logging.error(f"[utils_data_preparation] check_and_create_file : {e}")


def read_yolo_label_file(file_path):
    try:
        data = []

        with open(file_path, "r") as file:
            lines = file.readlines()

        if not lines:  ### Handle no labels
            data.append({"class_name": 999})

        for line in lines:
            values = line.strip().split()
            class_number = int(values[0])
            data.append({"class_name": class_number})  # , 'bbox': bbox})

        return data
    except Exception as e:
        logging.error(f"[utils_data_preparation] read_yolo_label_file : {e}")


def get_data(TASK, MAIN_DATA_FOLDER):
    try:
        df = None

        if TASK == "CLASSIFICATION":
            image_paths = []
            folder_names = []

            for root, dirs, files in os.walk(MAIN_DATA_FOLDER):
                for file in files:

                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".avif")
                    ):
                        image_path = os.path.join(root, file)
                        image_paths.append(image_path)

                        folder_name = os.path.basename(root)
                        folder_names.append(folder_name)

            df = pd.DataFrame(data={"path": image_paths, "class_name": folder_names})
            df["fold"] = -1

        elif (TASK == "DETECTION") | (TASK == "INSTANCE_SEGMENTATION"):
            all_data = []
            extension_list = []
            root_label_path = os.path.join(MAIN_DATA_FOLDER, "train", "labels")
            for root, dirs, files in os.walk(MAIN_DATA_FOLDER):
                for file in files:

                    # if file.lower().endswith((".txt")):
                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".avif")
                    ):
                        file_name, extension = os.path.splitext(file)
                        label_path = os.path.join(root_label_path, file_name + ".txt")

                        check_and_create_file(label_path)
                        data = read_yolo_label_file(label_path)

                        for entry in data:
                            entry["path"] = label_path.split(".txt")[0]
                            entry["img_extension"] = extension
                            entry["label_extension"] = ".txt"

                        all_data.extend(data)
            df = pd.DataFrame(all_data)
            
        return df
    except Exception as e:
        logging.error(f"[utils_data_preparation] get_data : {e}")


def split_data(df, kwargs):
    try:
        TASK = kwargs["task"]
        NO_FOLD = kwargs["no_fold"]
        # SPLIT_SEED = kwargs['split_seed']
        IS_SPLIT_TEST = kwargs["is_split_test"]
        target_column = "class_name"
        groups_column = "path"

        stratified_group_kfold = StratifiedGroupKFold(
            n_splits=NO_FOLD, shuffle=False
        )  # , random_state=SPLIT_SEED)
        for fold, (train_index, val_index) in enumerate(
            stratified_group_kfold.split(df, df[target_column], df[groups_column])
        ):
            df.loc[val_index, "fold"] = fold

        if TASK == "CLASSIFICATION":
            last_fold = df.fold.max()

            if IS_SPLIT_TEST:
                df["group"] = df["fold"].replace(last_fold, "test")
                df["group"] = df["group"].apply(lambda x: "train" if x != "test" else x)
            else:
                df["group"] = "train"

        return df

    except Exception as e:
        logging.error(f"[utils_data_preparation] split_data : {e}")


class CreateFolder:
    def __init__(self, kwargs) -> None:
        try:
            # self.POC_EXPERIMENT_PATH = os.path.join("assets", "poc_experiments") # ? should to get data on config
            self.POC_EXPERIMENT_PATH = kwargs["poc_experiments_path"]
            self.folder_base_name = None
            self.new_folder_name = None

        except Exception as e:
            logging.error(f"[utils_data_preparation] - [CreateFolder] __init : {e}")

    def __check_existing_folders(self):
        try:
            existing_folders = []
            # Iterate through each item in the current directory
            for folder in os.scandir(self.POC_EXPERIMENT_PATH):
                # Check if the item is a directory and its name starts with the specified base name
                if folder.is_dir() and folder.name.startswith(self.folder_base_name):
                    # Add the folder name to the list
                    existing_folders.append(folder.name)
            # print(existing_folders)
            return existing_folders

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreateFolder] __check_existing_folders : {e}"
            )

    def __check_highest_number(self, existing_folders):
        try:
            numeric_parts = []

            # Iterate through each folder name in the existing_folders list
            for folder in existing_folders:
                # Extract the numeric part of the folder name (after the folder_base_name)
                numbers = re.findall(r"\d+", folder)

                # Convert the extracted integers from strings to integers
                integers = [int(num) for num in numbers][0]
                # Convert the numeric part to an integer and add it to the list
                numeric_parts.append(int(integers))

            # print(max(numeric_parts, default=0))
            # Find the highest number from the numeric_parts list; default to 0 if the list is empty
            return max(numeric_parts, default=0)

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreateFolder] __check_highest_number : {e}"
            )

    def __create_parent_folder(self, TASK="DEFAULT"):
        try:
            self.folder_base_name = TASK
            existing_folders = self.__check_existing_folders()
            highest_number = self.__check_highest_number(existing_folders)
            self.new_folder_name = f"{self.folder_base_name}{highest_number + 1}"
            print(f"parent folder : {self.new_folder_name}")
            os.makedirs(
                os.path.join(self.POC_EXPERIMENT_PATH, self.new_folder_name),
                exist_ok=True,
            )

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreateFolder] __create_parent_folder : {e}"
            )

    def __create_child_foler(self, df, NO_FOLD, TASK="DEFAULT"):
        try:
            # root = os.path.join("assets", "poc_experiments", self.new_folder_name)
            root = os.path.join(self.POC_EXPERIMENT_PATH, self.new_folder_name)

            # create child by folds
            for idx in range(NO_FOLD):
                if TASK == "CLASSIFICATION":
                    class_name_arr = df.class_name.unique().tolist()
                    for cls in class_name_arr:

                        os.makedirs(
                            os.path.join(root, f"fold{idx}", "train", f"{cls}"),
                            exist_ok=True,
                        )
                        os.makedirs(
                            os.path.join(root, f"fold{idx}", "test", f"{cls}"),
                            exist_ok=True,
                        )

                elif (TASK == "DETECTION") | (TASK == "INSTANCE_SEGMENTATION"):
                    os.makedirs(
                        os.path.join(root, f"fold{idx}", "images"), exist_ok=True
                    )
                    os.makedirs(
                        os.path.join(root, f"fold{idx}", "labels"), exist_ok=True
                    )

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreateFolder] __create_child_foler : {e}"
            )

    def create_folder(self, df, kwargs):
        try:
            NO_FOLD = kwargs["no_fold"]
            TASK = kwargs["task"]
            self.__create_parent_folder(TASK)
            self.__create_child_foler(df, NO_FOLD, TASK)

            return self.new_folder_name

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreateFolder] create_folder : {e}"
            )


# copy data
def generate_combinations(data, no_folds):
    try:
        all_combinations = list(combinations(data, no_folds))
        combinations_dicts = []
        for comb in all_combinations:
            train_data = list(comb)
            val_data = [item for item in data if item not in comb]
            combinations_dicts.append({"train": train_data, "val": val_data})

        return combinations_dicts

    except Exception as e:
        logging.error(f"[utils_data_preparation] generate_combinations : {e}")


def copy_data_2_poc_experiments(df, new_folder_name, kwargs):
    try:
        # root = os.path.join("assets", "poc_experiments", new_folder_name)
        POC_EXPERIMENTS_PATH = kwargs["poc_experiments_path"]
        print("copy data 2 folder: ", POC_EXPERIMENTS_PATH)
        root = os.path.join(POC_EXPERIMENTS_PATH, new_folder_name)

        TASK = kwargs["task"]

        if TASK == "CLASSIFICATION":
            df_train = df[df.group == "train"]
            df_test = df[df.group == "test"]

            last_fold = df_train.fold.max()
            fold_list = list(range(last_fold + 1))
            combinations_dicts = generate_combinations(fold_list, len(fold_list) - 1)

            ### copy test data
            for file_name, class_name, fold in zip(
                df_test["path"].tolist(),
                df_test["class_name"].tolist(),
                df_test["fold"].tolist(),
            ):
                destination = os.path.join(root, f"fold{int(fold)}", "test", class_name)
                shutil.copy(file_name, destination)
                # print(destination)

            ### copy train data
            for idx, dataset_combination in enumerate(combinations_dicts):
                # print(dataset_combination)
                for file_name, class_name, fold in zip(
                    df_train["path"].tolist(),
                    df_train["class_name"].tolist(),
                    df_train["fold"].tolist(),
                ):
                    if fold in dataset_combination["train"]:
                        destination = os.path.join(
                            root, f"fold{int(idx)}", "train", class_name
                        )
                        shutil.copy(file_name, destination)
                        # print(destination)

                    elif fold in dataset_combination["val"]:
                        destination = os.path.join(
                            root, f"fold{int(idx)}", "test", class_name
                        )
                        shutil.copy(file_name, destination)
                        # print(destination)

        elif (TASK == "DETECTION") | (TASK == "INSTANCE_SEGMENTATION"):
            for file_name, class_name, fold, img_extension in zip(
                df["path"].tolist(),
                df["class_name"].tolist(),
                df["fold"].tolist(),
                df["img_extension"].tolist(),
            ):

                img_file_name = file_name.replace("labels", "images") + img_extension
                labels_file_name = file_name + ".txt"

                img_destination = os.path.join(root, f"fold{int(fold)}", "images")
                label_destination = os.path.join(root, f"fold{int(fold)}", "labels")

                shutil.copy(img_file_name, img_destination)
                shutil.copy(labels_file_name, label_destination)

    except Exception as e:
        logging.error(f"[utils_data_preparation] copy_data_2_poc_experiments : {e}")


class CreatePointerFile:
    def __init__(self) -> None:
        try:
            self.kwargs = None
            self.new_folder_name = None
            self.root = None
        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreatePointerFile] __init : {e}"
            )

    def __rename_folder(self, old_name, new_name):
        try:
            old_name = os.path.join(self.root, old_name)
            new_name = os.path.join(self.root, new_name)
            os.rename(old_name, new_name)
            print(f"Folder '{old_name}' has been renamed to '{new_name}'.")

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreatePointerFile] __rename_folder : {e}"
            )

    def __generate_combinations(self, data, len_data):
        try:
            all_combinations = list(combinations(data, len_data))

            result_dicts = []
            for comb in all_combinations:
                train_data = list(comb)
                val_data = [item for item in data if item not in comb]
                result_dicts.append({"train": train_data, "val": val_data})

            return result_dicts

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreatePointerFile] __generate_combinations : {e}"
            )

    def __create_poiter_test(self):
        try:
            destination = os.path.join(self.root, f"test.yaml")

            data = {
                # 'path' : os.path.join("..", self.root),
                "path": os.path.join(os.getcwd(), self.root),
                "train": None,
                "val": os.path.join("test", "images"),  # 128 images
                "nc": self.kwargs["nc"],
                "names": self.kwargs["class_dict"],
            }
            with open(destination, "w") as file:
                yaml.dump(data, file, default_flow_style=False)

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreatePointerFile] __create_poiter_test : {e}"
            )

    def create_pointer_file(self, new_folder_name, kwargs):
        try:
            self.kwargs = kwargs
            self.new_folder_name = new_folder_name
            self.POC_EXPERIMENTS_PATH = kwargs["poc_experiments_path"]
            self.root = os.path.join(self.POC_EXPERIMENTS_PATH, self.new_folder_name)
            TASK = self.kwargs["task"]

            if TASK == "CLASSIFICATION":  # Just rename
                list_folder_fold = os.listdir(self.root)
                list_folder_fold = [
                    item for item in list_folder_fold if not item.startswith(".")
                ]
                list_folder_fold = sorted(list_folder_fold, key=lambda x: int(x[4:]))
                print(list_folder_fold)
                if self.kwargs["is_split_test"]:
                    folder_test = list_folder_fold[-1]
                    print(list_folder_fold)
                    print(folder_test)
                    self.__rename_folder(folder_test, new_name="test")

            elif (TASK == "DETECTION") | (
                TASK == "INSTANCE_SEGMENTATION"
            ):  # create pointer & rename

                list_folder_fold = os.listdir(self.root)
                list_folder_fold = [
                    item for item in list_folder_fold if not item.startswith(".")
                ]
                list_folder_fold = sorted(list_folder_fold, key=lambda x: int(x[4:]))
                if self.kwargs["is_split_test"]:
                    folder_test = list_folder_fold[-1]
                    list_folder_fold.pop(-1)
                    self.__rename_folder(folder_test, new_name="test")

                flat_folders_with_images = [
                    os.path.join(folder, "images") for folder in list_folder_fold
                ]
                result = self.__generate_combinations(
                    flat_folders_with_images, len(flat_folders_with_images) - 1
                )

                for idx, fold_name in enumerate(list_folder_fold):

                    destination = os.path.join(self.root, f"{fold_name}.yaml")

                    data = {
                        # 'path' : os.path.join("..", self.root),
                        "path": os.path.join(os.getcwd(), self.root),
                        "train": result[idx]["train"],
                        "val": result[idx]["val"],  # 128 images
                        "nc": self.kwargs["nc"],
                        "names": self.kwargs["class_dict"],
                    }

                    with open(destination, "w") as file:
                        yaml.dump(data, file, default_flow_style=False)

                if self.kwargs["is_split_test"]:
                    self.__create_poiter_test()

        except Exception as e:
            logging.error(
                f"[utils_data_preparation] - [CreatePointerFile] create_pointer_file : {e}"
            )
