import open3d as o3d
import numpy as np
import os
import glob
import random

def visualize_colored_objects(frame_dir):
    """
    Visualizes all point cloud objects in a given frame directory,
    each painted with a random color.
    """
    # Grab all object_*.pcd files (ignore non-object files like mask or rgb)
    pcd_files = glob.glob(os.path.join(frame_dir, 'object_*.pcd'))

    if not pcd_files:
        print(f"No point clouds found in {frame_dir}")
        return

    geometries = []

    for pcd_file in pcd_files:
        # Load point cloud
        pcd = o3d.io.read_point_cloud(pcd_file)
        if not pcd.has_points():
            continue

        # Assign a random color to each object
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)

        geometries.append(pcd)

    # Launch interactive viewer
    print(f"Visualizing {len(geometries)} objects from: {frame_dir}")
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    # Replace this with your actual frame folder path
    FRAME_DIR = '/home/viny/Desktop/CLIP_Search/data_from_runs/data_1743688766_645412206/(1743688766, 719385623)/'

    # Normalize path if it's in parentheses
    #FRAME_DIR = FRAME_DIR.strip().replace("(", "").replace(")", "").replace(",", "_")

    # Run visualization
    visualize_colored_objects(FRAME_DIR)
