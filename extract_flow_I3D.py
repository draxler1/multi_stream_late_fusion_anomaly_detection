
''' This PyTorch code assumes that the optical flow features are stored in a numpy file 
called optical_flow_features, and that the I3D model has been imported from the
torchvision.models.i3d module.  The code initializes the I3D model with flow modality,
sets the model to evaluation mode,  and extracts the features from the mixed 5C layer. 
Finally, the code saves the features to a file.

NOTE: You can use i3d_feature extraction code also to extract I3D flow features. Don't forget to change the modality to flow.'''

import torch
import torchvision
import numpy

# Load the optical flow features
flow_features = numpy.load('optical_flow_features.pt')

# Initialize the I3D model with flow modality
i3d = torchvision.models.i3d.I3D(num_classes=400, modality='flow')

# Set the model to evaluation mode
i3d.eval()

# Extract features from the mixed 5C layer
with torch.no_grad():
  mixed_5c_features = i3d.mixed_5c(flow_features)

# Save the I3D features
numpy.save(mixed_5c_features, 'i3d_features_flow.npy')
