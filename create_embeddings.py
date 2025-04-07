import os
import torch
import cv2
import numpy as np
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Loaded CLIP onto device: {device}")


def random_perspective(image, max_warp=0.1):
    """
    Apply a small random perspective transform to an image.

    Args:
        image (np.ndarray): Input image.
        max_warp (float): Maximum warp as a fraction of image size (0.1 = 10%).

    Returns:
        np.ndarray: Warped image.
    """
    if type(image) != np.ndarray:
        image = np.array(image)
    h, w = image.shape[:2]
    # Define perturbation scale
    dx = int(w * max_warp)
    dy = int(h * max_warp)
    # Source points near the corners
    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])
    # Perturb the source points randomly
    dst_pts = np.float32([
        [np.random.randint(0, dx), np.random.randint(0, dy)],
        [w - 1 - np.random.randint(0, dx), np.random.randint(0, dy)],
        [w - 1 - np.random.randint(0, dx), h - 1 - np.random.randint(0, dy)],
        [np.random.randint(0, dx), h - 1 - np.random.randint(0, dy)]
    ])
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the image
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT101)

    return warped

def embed_image(image_path, model, preprocess):
    image = Image.open(image_path)
    images = [random_perspective(image) for _ in range(10)]
    images = [preprocess(Image.fromarray(image)) for image in images]
    images = torch.stack(images, dim = 0).to(device)
    with torch.no_grad():
        image_embeddings = model.encode_image(images).cpu()
    return torch.mean(image_embeddings, dim=0).numpy()

def create_image_ebeddings(images_path = "./images"):
    os.makedirs("./embeddings", exist_ok=True)
    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path, image_name)
        image_embedding = embed_image(image_path, model, preprocess)
        np.save(f"./embeddings/{image_name[:-4]}.npy", image_embedding)

if __name__ == "__main__":
    create_image_ebeddings()
