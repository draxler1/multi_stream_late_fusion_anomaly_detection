import os
import cv2
import torch
import numpy as np

# SelFlow model trained on the Sintel dataset
# Source: https://github.com/ClementPinard/SintelFlow
model = torch.load("SelFlow.pth")

# Set model to evaluation mode
model.eval()

# Read video file
cap = cv2.VideoCapture("video.mp4")

# Extract video metadata
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize output array to store features of each segment
segment_count = frame_count // 32  # Number of segments
features = np.zeros((segment_count, 2*1024))

# Process video frame by frame
segment_idx = 0
clip_idx = 0
prev_frame = None
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    frame = cv2.resize(frame, (256, 192))
    frame = frame.astype(np.float32)
    frame -= np.array([[[0.5, 0.5, 0.5]]])
    frame /= np.array([[[0.5, 0.5, 0.5]]])
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    frame = torch.from_numpy(frame)
    
    # Extract optical flow features using SelFlow model
    if prev_frame is not None:
        with torch.no_grad():
            output = model(prev_frame, frame)
            output = output.squeeze(dim=0)
            features[segment_idx, :] += output.numpy()
    
    # Check if we have processed 16 frames (1 clip)
    clip_idx += 1
    if clip_idx == 16:
        clip_idx = 0
        segment_idx += 1

# Average features over the segment
features /= 16

# Close video file
cap.release()

# Save features to file
np.save("flow.npy")
