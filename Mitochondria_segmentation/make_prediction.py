import sys, os, cv2, csv
import pandas as pd
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from skimage.measure import regionprops, label
from ultralytics import YOLO
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from time import time

# ________________ HANDLE THE PATH THING ________________ #
# get the absolute path of the script's directory
script_path = os.path.dirname(os.path.abspath(__file__))
# get the parent directory of the script's directory
parent_path = os.path.dirname(script_path)
sys.path.append(parent_path)
from utils import *
from Mitochondria_segmentation.path_config import PathConfiguration


# ________________ MAKE MODEL CONFIGURATION DETECTRON2________________ #
class Detectron2Configuration:
    def __init__(self, chosen_model, thresh_score):
        self.chosen_model = chosen_model
        self.thresh_score = thresh_score
        self.path_to_train_JSON = PathConfiguration.path_to_train_JSON
        self.path_to_train_images = PathConfiguration.path_to_train_images
        self.path_to_chosen_model = PathConfiguration.path_to_chosen_model_detectron2
        self.cfg = get_cfg()

    def make_configuration(self):
        try:
            # ________________ DATA REGISTRATION FOR PREDICTION ________________ #
            # Check if the dataset is already registered
            if "train" not in DatasetCatalog.list():
                register_coco_instances(
                    "train", {}, self.path_to_train_JSON, self.path_to_train_images
                )
            train_metadata = MetadataCatalog.get("train")
            train_dataset_dicts = DatasetCatalog.get("train")

            # ________________ CONFIGURATION SETUP FOR PREDICTION ________________ #
            cfg = self.cfg
            chosen_model = self.chosen_model
            number_classes = get_num_classes(str(train_metadata))

            cfg.merge_from_file(model_zoo.get_config_file(chosen_model))
            cfg.MODEL.WEIGHTS = os.path.join(
                self.path_to_chosen_model, "model_final.pth"
            )
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.thresh_score
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes
            
            # Set model to use CPU in torch
            cfg.MODEL.DEVICE = 'cpu'

            # Create predictor & metadata
            predictor = DefaultPredictor(cfg)
            metadata = train_metadata

            return predictor, metadata

        except Exception as e:
            raise CustomException(e, sys)


# ________________ MAKE SEGMENTATION DETECTRON2 ________________ #
class Detectron2Segmentation:
    def __init__(self, predictor, metadata):
        self.path_to_images = PathConfiguration.path_to_images_input
        self.path_to_save_segm = PathConfiguration.path_to_images_output
        self.predictor = predictor
        self.metadata = metadata
        self.image_id = int(time())

    # ________________ MAKE PREDICTION IMAGES & CSV ________________ #
    def make_prediction(self):
        try:
            if not os.path.exists(self.path_to_save_segm):
                os.makedirs(self.path_to_save_segm)

            csv_file_name = f"{self.image_id}.csv"

            full_path_to_save_segm_csv = os.path.join(
                self.path_to_save_segm, csv_file_name
            )

            # Open the CSV file for writing
            with open(full_path_to_save_segm_csv, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)

                # Write the header row in the CSV file
                csv_writer.writerow(
                    [
                        "File Name",
                        "Image ID",
                        "Class Name",
                        "Object Number",
                        "Area",
                        "Centroid",
                        "BoundingBox",
                    ]
                )  # Add more columns as needed for other properties

                # Initialize counters
                total_objects = 0
                total_area = 0
                # Loop over the images in the input folder
                for image_file_name in os.listdir(self.path_to_images):
                    # get the original file extension (e.g., .jpg, .png)
                    file_ext = os.path.splitext(image_file_name)[1]
                    # generate a unique filename by appending a timestamp to the original filename
                    unique_image_name = f"{self.image_id}{file_ext}"
                    image_id = unique_image_name

                    full_path_to_image = os.path.join(
                        self.path_to_images, image_file_name
                    )
                    image = cv2.imread(full_path_to_image)

                    # Perform prediction on the new image
                    segm_image = self.predictor(image)

                    # ________ MAKE CSV SEGMENTATION _______ #
                    # Convert the predicted mask to a binary mask
                    mask = (
                        segm_image["instances"]
                        .pred_masks.to("cpu")
                        .numpy()
                        .astype(bool)
                    )

                    if mask.size == 0:
                        error_message = f"No objects detected in {image_file_name}"
                        os.remove(full_path_to_image)
                        os.remove(full_path_to_save_segm_csv)

                    else:
                        # Get the predicted class labels
                        class_labels = (
                            segm_image["instances"].pred_classes.to("cpu").numpy()
                        )

                        # Use skimage.measure.regionprops to calculate object parameters
                        labeled_mask = label(mask)
                        props = regionprops(labeled_mask)

                    # Write the object-level information to the CSV file
                    for i, prop in enumerate(props):
                        object_number = i + 1  # Object number starts from 1
                        area = prop.area
                        centroid = prop.centroid
                        bounding_box = prop.bbox

                        # Check if the corresponding class label exists
                        if i < len(class_labels):
                            class_label = class_labels[i]
                            class_name = self.metadata.thing_classes[class_label]
                        else:
                            # If class label is not available (should not happen), use 'Unknown' as class name
                            class_name = "Unknown"

                        # Write the object-level information to the CSV file
                        csv_writer.writerow(
                            [
                                image_file_name,
                                image_id,
                                class_name,
                                object_number,
                                area,
                                centroid,
                                bounding_box,
                            ]
                        )  # Add more columns as needed for other properties

                    # ________ MAKE IMAGE SEGMENTATION _______ #
                    # Use `Visualizer` to draw the predictions on the image.
                    image_visual = Visualizer(image[:, :, ::-1], metadata=self.metadata)
                    visual_out = image_visual.draw_instance_predictions(
                        segm_image["instances"].to("cpu")
                    )

                    # Create the output file_name with _result extension
                    result_file_name = os.path.splitext(image_id)[0] + f"{file_ext}"
                    path_to_save_segm_images_full = os.path.join(
                        self.path_to_save_segm, result_file_name
                    )

                    # Save the segmented image
                    cv2.imwrite(
                        path_to_save_segm_images_full,
                        visual_out.get_image()[:, :, ::-1],
                    )

                    # Update counters
                    total_objects = len(props)
                    total_area += sum([prop.area for prop in props])
                    # Calculate mean area
                    mean_area = total_area / total_objects if total_objects > 0 else 0

                    # remove the just got predicted image
                    os.remove(full_path_to_image)

            return total_objects, mean_area, image_id, csv_file_name

        except Exception as e:
            raise CustomException(e, sys)


# ________________ MAKE SEGMENTATION DETECTRON2 ________________ #
class YOLOv8Segmentation:
    def __init__(self, conf=0.25):
        self.path_to_chosen_model = PathConfiguration.path_to_chosen_model_YOLOv8
        self.path_to_images = PathConfiguration.path_to_images_input
        self.path_to_save_segm = PathConfiguration.path_to_images_output
        self.path_to_save_segm_CSV = PathConfiguration.path_to_images_output
        self.image_id = int(time())
        self.conf = conf

    # ________________ MAKE PREDICTION IMAGES & CSV ________________ #
    def make_prediction(self):
        try:
            # Create the output directory if it doesn't exist
            if not os.path.exists(self.path_to_save_segm):
                os.makedirs(self.path_to_save_segm)

            if not os.path.exists(self.path_to_save_segm_CSV):
                os.makedirs(self.path_to_save_segm_CSV)

            model = YOLO(self.path_to_chosen_model)

            # Process each image file in the input directory
            for image_file_name in os.listdir(self.path_to_images):
                # Check for image file extensions
                if not image_file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                # get the original file extension (e.g., .jpg, .png)
                file_ext = os.path.splitext(image_file_name)[1]
                # generate a unique filename by appending a timestamp to the original filename
                unique_image_name = f"{self.image_id}{file_ext}"
                image_id = unique_image_name

                # Full path to the image file
                full_path_to_image = os.path.join(self.path_to_images, image_file_name)

                # ________ MAKE IMAGE SEGMENTATION _______ #
                prediction = model.predict(full_path_to_image, conf=self.conf)

                prediction_array = prediction[0].plot()

                # Create a figure and axis to display the image
                fig, ax = plt.subplots(1)
                ax.imshow(prediction_array)
                ax.axis("off")  # Hide the axes

                # Create the output file_name with _result extension
                result_file_name = os.path.splitext(image_id)[0] + f"{file_ext}"
                full_path_to_save_segm_images = os.path.join(
                    self.path_to_save_segm, result_file_name
                )

                plt.savefig(
                    full_path_to_save_segm_images, bbox_inches="tight", pad_inches=0
                )
                plt.close(fig)

                prediction_0 = prediction[0]
                # print(len(prediction_0.masks))

                if not prediction_0.masks:
                    os.remove(full_path_to_image)
                    os.remove(full_path_to_save_segm_images)

                else:
                    # ________ MAKE CSV SEGMENTATION _______ #
                    extracted_masks = prediction_0.masks.data
                    # Extract the boxes, which likely contain class IDs
                    detected_boxes = prediction_0.boxes.data
                    # Extract class IDs from the detected boxes
                    class_labels = detected_boxes[:, -1].int().tolist()
                    masks_by_class = {name: [] for name in prediction_0.names.values()}

                    # Iterate through the masks and class labels
                    for mask, class_id in zip(extracted_masks, class_labels):
                        class_name = prediction_0.names[
                            class_id
                        ]  # Map class ID to class name
                        masks_by_class[class_name].append(mask.cpu().numpy())

                    # Initialize a list to store the properties
                    props_list = []

                    # Iterate through all classes
                    for class_name, masks in masks_by_class.items():
                        # Iterate through the masks for this class
                        for mask in masks:
                            # Convert the mask to an integer type if it's not already
                            mask = mask.astype(int)

                            # Apply regionprops to the mask
                            props = regionprops(mask)

                            # Extract the properties you want (e.g., area, perimeter) and add them to the list
                            for prop in props:
                                area = prop.area
                                perimeter = prop.perimeter
                                # Add other properties as needed

                                # Append the properties and class name to the list
                                props_list.append(
                                    {
                                        "Class_Name": class_name,
                                        "Area": area,
                                        "Perimeter": perimeter,
                                    }
                                )

                    # Convert the list of dictionaries to a DataFrame
                    props_df = pd.DataFrame(props_list)
                    # Save the DataFrame to a CSV file
                    csv_file_name = os.path.splitext(image_id)[0]
                    full_path_to_save_segm_CSV = os.path.join(
                        self.path_to_save_segm_CSV, f"{csv_file_name}.csv"
                    )
                    props_df.to_csv(full_path_to_save_segm_CSV, index=False)

                    total_objects = len(props_df.index)
                    total_area = props_df["Area"].sum()
                    mean_area = total_area / total_objects if total_objects > 0 else 0

                    # remove the just got predicted image
                    os.remove(full_path_to_image)

                return total_objects, round(mean_area, 2), image_id, csv_file_name

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     # # **** --------- **** #
#     # chosen_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
#     # thresh_score = 0.5
#     # # **** --------- **** #

#     # ModelConfigurator = ModelConfiguration(chosen_model, thresh_score)
#     # predictor, metadata = ModelConfigurator.make_configuration()
#     # MitoSegmentator = MitoSegmentation(predictor, metadata)
#     # MitoSegmentator.make_prediction()

#     YOLOv8Segmentator = YOLOv8Segmentation()
#     (
#         total_objects,
#         mean_area,
#         image_id,
#         csv_file_name,
#     ) = YOLOv8Segmentator.make_prediction()

#     # YOLOv8Segmentator.make_prediction()

#     print(total_objects, mean_area, image_id, csv_file_name)
