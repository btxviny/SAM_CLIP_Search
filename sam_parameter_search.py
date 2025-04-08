import clip
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from segment_anything import SamAutomaticMaskGenerator, build_sam
from pathlib import Path
import mlflow
import yaml
from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model, preprocess = clip.load("ViT-B/32", device=device)  # Load CLIP on GPU

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

@torch.no_grad()
def get_masks(image, mask_generator):
    return mask_generator.generate(image)

def get_bboxes(image, masks):
    cropped_boxes = []
    for mask in masks:
        cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))
    return cropped_boxes

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", segmented_image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

@torch.no_grad()
def compute_clip_features_in_batches(images, batch_size=16):
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        stacked = torch.stack(batch).to(device)
        features = model.encode_image(stacked)
        features /= features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())  # Move to CPU to free up GPU memory
        torch.cuda.empty_cache()  # Free up memory
    return torch.cat(all_features, dim=0).to(device)  # Move back to GPU if needed

@torch.no_grad()
def retriev(image_features, text_features):
    # Compute similarity scores between precomputed image features and text features
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

def get_semantic_match(image, search_text, masks, image_features):
    # Tokenize and compute embedding for the search text
    tokenized_text = clip.tokenize([search_text]).to(device)  # Move text to the GPU
    text_features = model.encode_text(tokenized_text)
    
    # Retrieve matching scores for the cropped images using precomputed image features
    scores = retriev(image_features, text_features)
    
    # Get indices of the masks that exceed the threshold
    indices = get_indices_of_values_above_threshold(scores, 0.05)
    
    # Prepare the segmentation masks to overlay on the original image
    segmentation_masks = []
    for seg_idx in indices:
        segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
        segmentation_masks.append(segmentation_mask_image)

        # Create overlay for visualization
        original_image = Image.fromarray(image)
        overlay_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
        overlay_color = (255, 0, 0, 100)

        draw = ImageDraw.Draw(overlay_image)
        for segmentation_mask_image in segmentation_masks:
            draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)
    
    # Combine original image with overlay
    result = np.array(Image.alpha_composite(original_image.convert('RGBA'), overlay_image))
    
    # Add the search query text to the result image
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
    logger.info("Running")
    # Load configuration from YAML file
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    
    images = config_data["images"]
    queries = config_data["queries"]
    sam_configs = config_data["sam_configs"]

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("semantic_retrieval_sam_v6")
    
    logger.info("Starting experiments for SAM parameter tuning...")
    
    # Loop through each SAM configuration
    for sam_config in sam_configs:
        logger.info(f"Starting MLflow run for configuration: {sam_config['name']}")
        with mlflow.start_run(run_name=sam_config["name"]):
            mlflow.log_params(sam_config)
            logger.info(f"Logged SAM config parameters to MLflow: {sam_config}")

            # Build SAM model and initialize the mask generator with current parameters
            sam_model = build_sam(checkpoint="sam_vit_h_4b8939.pth")
            # Ensure SAM is on CPU
            sam_model.to("cpu").eval()  # Ensure SAM stays on CPU
            
            mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=sam_config["points_per_side"],
                pred_iou_thresh=sam_config["pred_iou_thresh"],
                stability_score_thresh=sam_config["stability_score_thresh"],
                box_nms_thresh=sam_config["box_nms_thresh"],
                crop_n_layers=sam_config["crop_n_layers"],
                min_mask_region_area=sam_config["min_mask_region_area"]
            )
            logger.info(f"Initialized mask generator for configuration: {sam_config['name']}")

            # Loop through images
            for image_path in images:
                logger.info(f"Processing image: {image_path}")
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Compute masks and cropped boxes once for the image
                logger.info(f"Computing masks and cropped boxes for image: {image_path}")
                masks = get_masks(image, mask_generator)
                cropped_boxes = get_bboxes(image, masks)

                # Precompute the image features for all cropped boxes for the current image
                torch.cuda.empty_cache()  # Clear GPU memory before processing
                preprocessed_cropped_boxes = [preprocess(cropped_box) for cropped_box in cropped_boxes]
                logger.info(f"Computing CLIP features for image in batches: {image_path}")
                image_features = compute_clip_features_in_batches(preprocessed_cropped_boxes, batch_size=8)


                # Process each query for the image
                for query in queries:
                    logger.info(f"Processing query: {query}")
                    result_image = get_semantic_match(image, query, masks, image_features)
                    
                    # Generate a unique name for this result
                    result_name = f"{sam_config['name']}_{Path(image_path).stem}_{query.replace(' ', '_')}"
                    log_image_to_mlflow(result_image, result_name)
                    logger.info(f"Logged result image: {result_name}")

    logger.info("Finished all experiments.")
