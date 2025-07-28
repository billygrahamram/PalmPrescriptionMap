import os
import json
import time
import gc
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import os

# Get the current working directory (where the script is run from)
current_directory = os.getcwd()
print(f"The script is running from: {current_directory}")

# Correct string interpolation for paths
model_paths = [os.path.join(f"{current_directory}/models/yolo11n-seg_bgr.engine")]
orthophoto_path = [os.path.join(f"{current_directory}/output/project_1/datasets/project/odm_orthophoto/odm_orthophoto.png")]
export_dir = os.path.join(f"{current_directory}")
tile_size = 640
CONF_THRESH = 0.45


def slice_image(img, tile_size):
    slices = []
    h, w, _ = img.shape
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]
            padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded_tile[:tile.shape[0], :tile.shape[1]] = tile
            slices.append((x, y, padded_tile, tile.shape[1], tile.shape[0]))
    return slices

def run_inference_on_slices(model, slices, image_shape):
    all_detections = []
    mask_canvas = np.zeros(image_shape[:2], dtype=np.uint8)

    total_pre, total_inf, total_post = 0, 0, 0
    num_frames = 0

    for x_off, y_off, tile, tile_w, tile_h in slices:
        results = model.predict(tile, verbose=False, device="cuda:0")
        for result in results:
            total_pre += result.speed['preprocess']
            total_inf += result.speed['inference']
            total_post += result.speed['postprocess']
            num_frames += 1

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                for i, box in enumerate(result.boxes):
                    if box.conf < CONF_THRESH:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    x1_glob = int(x1 + x_off)
                    y1_glob = int(y1 + y_off)
                    x2_glob = int(x2 + x_off)
                    y2_glob = int(y2 + y_off)
                    all_detections.append((x1_glob, y1_glob, x2_glob, y2_glob, cls, conf))

                    mask = (masks[i] > 0.5).astype(np.uint8) * 255
                    mh, mw = mask.shape

                    # Clip mask region if tile exceeds image boundary
                    h_clip = min(mh, mask_canvas.shape[0] - y_off)
                    w_clip = min(mw, mask_canvas.shape[1] - x_off)
                    mask_canvas[y_off:y_off+h_clip, x_off:x_off+w_clip] = np.maximum(
                        mask_canvas[y_off:y_off+h_clip, x_off:x_off+w_clip],
                        mask[:h_clip, :w_clip]
                    )

    return all_detections, mask_canvas, total_pre/num_frames, total_inf/num_frames, total_post/num_frames, num_frames

def draw_boxes(image, detections):
    for x1, y1, x2, y2, cls, conf in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def generate_coco_annotations(detections, image_id, filename, width, height):
    coco = {
        "images": [{
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        }],
        "annotations": [],
        "categories": []
    }

    category_set = set()
    for idx, (x1, y1, x2, y2, cls, conf) in enumerate(detections):
        coco["annotations"].append({
            "id": idx,
            "image_id": image_id,
            "category_id": int(cls),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": (x2 - x1) * (y2 - y1),
            "iscrowd": 0,
            "score": conf
        })
        category_set.add(int(cls))

    for cid in sorted(category_set):
        coco["categories"].append({"id": cid, "name": str(cid), "supercategory": "none"})

    return coco

def write_system_info(f):
    f.write("\n========== System Information ==========\n")
    f.write(f"OS: {platform.system()} {platform.release()}\n")
    if torch.cuda.is_available():
        f.write("CUDA Available: Yes\n")
        f.write(f"Device Name: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA Capability: {torch.cuda.get_device_capability(0)}\n")
        f.write(f"Total VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)} GB\n")
    else:
        f.write("CUDA Available: No\n")

def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay a binary mask on the image with transparency."""
    color_mask = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        color_mask[:, :, c] = color[c]
    mask_bool = mask > 0
    image[mask_bool] = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)[mask_bool]
    return image

def run_inference(model_path, image_path):


    Path(export_dir).mkdir(parents=True, exist_ok=True)

    custom_filename = f"result"

    total_start = time.time()


    model = YOLO(model_path, task='segment')



    image = cv2.imread(image_path)


 
    slices = slice_image(image, tile_size)

    print(f"Generated {len(slices)} tiles")


    detections, mask_canvas, avg_pre, avg_inf, avg_post, num_frames = run_inference_on_slices(model, slices, image.shape)
    print(f"Found {len(detections)} total detections")
    


    # Optional: save raw mask separately if still needed
    cv2.imwrite(os.path.join(export_dir, f"{custom_filename}_mask.png"), mask_canvas)


    del model
    torch.cuda.empty_cache()
    gc.collect()


# Run inference
for img_path in orthophoto_path:
    for mdl_path in model_paths:
        print(f"Running {mdl_path} on {img_path}...")
        run_inference(mdl_path, img_path)
        print(f"Done.\n{'-' * 40}")
