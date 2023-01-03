import os
import cv2
import torch
import numpy as np

# I3D model trained on the Kinetics dataset
# Source: https://github.com/hassony2/kinetics_i3d_pytorch
model = torch.load("I3D.pth")

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
features = np.zeros((segment_count, 1024))

# Process video frame by frame
segment_idx = 0
clip_idx = 0
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32)
    frame -= np.array([[[123.68, 116.779, 103.939]]])
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    frame = torch.from_numpy(frame)
    
    # Extract features using I3D model
    with torch.no_grad():
        output = model(frame)
        output = output[4]  # mixed5c layer
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
np.save("i3d_features.npy", features)
