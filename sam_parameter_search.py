from pathlib import Path

import clip
import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from segment_anything import SamAutomaticMaskGenerator, build_sam, build_sam_vit_b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

@torch.no_grad()
def get_masks(image, mask_generator):
    return mask_generator.generate(image)


def get_bboxes(image_path, masks):
    image = Image.open(image_path)
    cropped_boxes = []
    for mask in masks:
        cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))
    return cropped_boxes

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image



@torch.no_grad()
def retriev(elements: list[Image.Image], search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


def get_semantic_match(image_path, search_text, mask_generator):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = get_masks(image,mask_generator)
    cropped_boxes = get_bboxes(image_path, masks)
    scores = retriev(cropped_boxes, search_text)
    indices = get_indices_of_values_above_threshold(scores, 0.05)
    segmentation_masks = []
    for seg_idx in indices:
        segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        segmentation_masks.append(segmentation_mask_image)

        original_image = Image.open(image_path)
        overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
        overlay_color = (255, 0, 0, 100)

        draw = ImageDraw.Draw(overlay_image)
        for segmentation_mask_image in segmentation_masks:
            draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)
    result = np.array(Image.alpha_composite(original_image.convert('RGBA'), overlay_image))
    cv2.putText(
        result,
        f"Query: {search_text}",
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 0, 0),  # Red in RGB
        thickness=2,
        lineType=cv2.LINE_AA
    )
    return Image.fromarray(result)

def log_image_to_mlflow(image: Image.Image, name: str):
    path = f"/tmp/{name}.png"
    image.save(path)
    mlflow.log_artifact(path, artifact_path="results")

if __name__ == "__main__":
   with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    images = config["images"]
    queries = config["queries"]
    sam_configs = config["sam_configs"]

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("semantic_retrieval_sam")
    
    # run in terminal to launch ui
    # mlflow server \
    # --backend-store-uri sqlite:///mlflow.db \
    # --default-artifact-root ./mlruns \
    # --host 0.0.0.0 \
    # --port 5001

    for config in sam_configs:
        with mlflow.start_run(run_name=config["name"]):
            mlflow.log_params(config)

            sam_model =  build_sam(checkpoint="sam_vit_h_4b8939.pth")
            mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=config["points_per_side"],
                pred_iou_thresh=config["pred_iou_thresh"],
                stability_score_thresh=config["stability_score_thresh"],
                box_nms_thresh=config["box_nms_thresh"],
                crop_n_layers=config["crop_n_layers"],
                min_mask_region_area=config["min_mask_region_area"]
            )
            for image_path in images:
                for query in queries:
                    result_image = get_semantic_match(image_path, query, mask_generator)

                    name = f"{config['name']}_{Path(image_path).stem}_{query.replace(' ', '_')}"
                    log_image_to_mlflow(result_image, name)