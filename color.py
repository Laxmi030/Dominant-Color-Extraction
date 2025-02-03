import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to extract dominant colors using KMeans
def extract_colors(image_path, k=5):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to RGB (OpenCV loads images in BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels (height * width, 3)
    pixels = img_rgb.reshape(-1, 3)
    
    # Use KMeans clustering to find the dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_
    
    # Convert colors to integer values
    colors = colors.round(0).astype(int)
    
    # Get the cluster labels for each pixel
    labels = kmeans.labels_
    
    # Count the number of pixels in each cluster (color)
    color_counts = [np.sum(labels == i) for i in range(k)]
    
    return colors, color_counts

# Function to plot the colors and their usage
def plot_colors(colors, color_counts):
    # Plot the dominant colors
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow([colors])
    plt.axis('off')
    plt.title("Dominant Colors")

    # Plot the bar graph for color usage
    plt.subplot(1, 2, 2)
    plt.bar(range(len(color_counts)), color_counts, color=colors/255)  # Normalize color values for display
    plt.xlabel('Color Index')
    plt.ylabel('Pixel Count')
    plt.title('Color Usage in Image')

    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    image_path = 'PROJECT/IMG_20220129_112936_296 - Copy.jpg'  # Replace with your image path
    k = 5  # Number of dominant colors to extract
    
    colors, color_counts = extract_colors(image_path, k)
    print("Dominant Colors (RGB):", colors)
    print("Color Usage (Pixel Count):", color_counts)
    
    plot_colors(colors, color_counts)
