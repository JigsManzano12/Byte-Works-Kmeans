# Video Clustering using Cluster.py

## Overview

Cluster.py is a Python script that clusters similar videos using KMeans clustering with features extracted from VGG16, and visualizes the clusters using PCA and Silhouette Analysis.

## Features

- **Video Clustering:** Groups videos based on visual similarities using deep features from VGG16.
- **Optimal K Determination:** Uses Silhouette Analysis to find the optimal number of clusters.
- **Visualization:** Generates visual plots of clusters using PCA for easy interpretation.
- **Incremental Learning:** Updates clusters dynamically with new videos.

## Requirements

- Python 3.x
- OpenCV (cv2)
- Keras (with TensorFlow backend)
- scikit-learn
- tqdm
- matplotlib

## Usage

1. **Setup:**
   - Ensure all dependencies are installed (`pip install -r requirements.txt`).
   - Make sure videos are organized in directories (`static/recorded_fights` for initial and new videos).

2. **Run Cluster.py:**
   -python Cluster.py

3. **Output:**
- Clusters are saved in `Model_Used/Clustered_Features/video_cluster_feature.npy`.
- Visualization plots saved in `static/Figures`.

## Details

### Extracting Frames and Features

- **extract_frames(video_path, num_frames=45):**
Extracts frames evenly spaced from videos using OpenCV.

- **extract_features(frames, model):**
Uses VGG16 pre-trained on ImageNet to extract deep features from frames.

### Clustering and Visualization

- **find_optimal_num_clusters_silhouette(features):**
Determines the optimal number of clusters using Silhouette Analysis.

- **visualize_clusters(features, clusters, save_path):**
Uses PCA to reduce dimensions and visualizes clusters in a 2D plot, saving it as an image.

### Incremental Learning and Deployment

- **Classifier Training:**
Uses KNeighborsClassifier to train on existing features and clusters.

- **Updating Clusters:**
Incorporates new videos, predicts their clusters, and updates existing clusters dynamically.

### Directories and Files

- **Directories:**
- `Model_Used/Video_Features/`: Stores extracted video features.
- `static/Figures/`: Saves visualizations.
- `static/Clusters/Cluster_{index}/`: Organizes videos by cluster after clustering.

- **Files:**
- `features_file`: Path to saved video features.
- `clusters_file`: Path to saved cluster assignments.
- `backup_dir`: Directory for backing up previous features and clusters.

## License

[Specify the license under which this script is distributed.]

## Author

Gerard Jose L Manzano

## Notes

- Ensure `static/recorded_fights` contains videos in supported formats (`mp4`, `avi`, `mov`).
- Adjust `num_frames` and other parameters in functions for different video qualities and clustering needs.

