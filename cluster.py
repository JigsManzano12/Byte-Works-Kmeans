import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Function to extract frames from a video
def extract_frames(video_path, num_frames=45):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_step = max(1, total_frames // num_frames)

    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frames_step)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

# Function to extract features from frames using VGG16
def extract_features(frames, model):
    features = []
    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x)[0])
    return np.array(features)

# Function to visualize clusters in a single plot
def visualize_clusters(features, clusters, save_path):
    num_components = min(features.shape[0], features.shape[1])  # Choose the minimum of samples and features
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(clusters))))  # Generate distinct colors for clusters
    for i, cluster_label in enumerate(np.unique(clusters)):
        cluster_indices = np.where(clusters == cluster_label)[0]
        cluster_pca_result = pca_result[cluster_indices]
        plt.scatter(cluster_pca_result[:, 0], cluster_pca_result[:, 1], label=f'Cluster {cluster_label}', color=colors[i])

    plt.title('KMeans Clustering of Videos')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.legend()
    plt.savefig(save_path)  # Save the plot as an image file
    plt.close()

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Define paths
features_file = 'Model_Used/Video_Features/all_video_features.npy'
clusters_file = 'Model_Used/Clustered_Features/video_cluster_feature.npy'
backup_dir = 'Model_Used/Backup/'
figure_dir = 'static/Figures'

# Ensure directories exist
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# Backup existing features and clusters
if os.path.exists(features_file) and os.path.exists(clusters_file):
    shutil.move(features_file, os.path.join(backup_dir, os.path.basename(features_file)))
    shutil.move(clusters_file, os.path.join(backup_dir, os.path.basename(clusters_file)))
    print("Old feature and cluster files moved to backup directory.")

# Load or extract and save initial video features
try:
    existing_features = np.load(features_file)
except FileNotFoundError:
    initial_video_dir = "static/recorded_fights"
    initial_video_features = []

    # Check if there are any video files in the directory
    if os.listdir(initial_video_dir):
        for video_file in tqdm(os.listdir(initial_video_dir)):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(initial_video_dir, video_file)
                frames = extract_frames(video_path)
                features = extract_features(frames, model)
                avg_features = np.mean(features, axis=0)
                initial_video_features.append(avg_features)
        existing_features = np.array(initial_video_features)
        np.save(features_file, existing_features)
    else:
        raise FileNotFoundError("No video files found in the initial video directory.")

# Determine optimal number of clusters using Silhouette Analysis
def find_optimal_num_clusters_silhouette(features):
    silhouette_scores = []
    K = range(4, 10)  # Try different values of K, starting from 4
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_labels = kmeans.fit_predict(features)
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    # Plot silhouette scores
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis For Optimal K')
    plt.savefig('static/Figures/silhouette_analysis.png')  # Save the plot as an image file
    plt.close()  # Close the plot
    # Find the optimal number of clusters using Silhouette Analysis
    optimal_num_clusters = np.argmax(silhouette_scores) + 4  # Add 4 because we started from K=4
    return optimal_num_clusters

# Apply KMeans Clustering with optimal number of clusters determined by Silhouette Analysis
num_clusters = find_optimal_num_clusters_silhouette(existing_features)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
existing_clusters = kmeans.fit_predict(existing_features)

# Save initial clusters
np.save(clusters_file, existing_clusters)

# Train a classifier on the existing features and clusters
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(existing_features, existing_clusters)

# Directory containing new videos
new_video_dir = "static/recorded_fights"

# Process each new video
new_video_features = []
for video_file in tqdm(os.listdir(new_video_dir)):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(new_video_dir, video_file)
        frames = extract_frames(video_path)
        features = extract_features(frames, model)
        avg_features = np.mean(features, axis=0)
        new_video_features.append(avg_features)

new_video_features = np.array(new_video_features)

# Predict clusters for the new features
new_clusters = classifier.predict(new_video_features)

# Update and save the new cluster assignments
combined_features = np.concatenate((existing_features, new_video_features))
combined_clusters = np.concatenate((existing_clusters, new_clusters))
np.save(features_file, combined_features)
np.save(clusters_file, combined_clusters)

# Visualize and save the updated clusters in a single plot
figure_path = 'static/Figures/kmeans_clusters.png'  # Specify the path to save the image
visualize_clusters(combined_features, combined_clusters, figure_path)

# Move videos to their respective cluster directories
video_filenames = [f for f in os.listdir(new_video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
cluster_dir = 'static/Clusters'

# Create a directory for each cluster if not already exist
for cluster_idx in range(np.max(combined_clusters) + 1):
    cluster_subdir = os.path.join(cluster_dir, f'Cluster_{cluster_idx}')
    os.makedirs(cluster_subdir, exist_ok=True)

# Move videos to their respective cluster directories
for video, cluster in zip(video_filenames, combined_clusters):
    src = os.path.join(new_video_dir, video)
    dst = os.path.join(cluster_dir, f'Cluster_{cluster}', video)
    shutil.copy(src, dst)

print("Processing and visualization completed.")
print("Videos have been moved to their respective cluster directories.")
