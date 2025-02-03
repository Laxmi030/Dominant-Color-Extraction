import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def extract_colors(image_path, k=5):
   
    img = cv2.imread(image_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pixels = img_rgb.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_

    colors = colors.round(0).astype(int)

    labels = kmeans.labels_
 
    color_counts = [np.sum(labels == i) for i in range(k)]
    
    return colors, color_counts

def plot_colors(colors, color_counts):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow([colors])
    plt.axis('off')
    plt.title("Dominant Colors")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(color_counts)), color_counts, color=colors/255)  
    plt.xlabel('Color Index')
    plt.ylabel('Pixel Count')
    plt.title('Color Usage in Image')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = 'PROJECT/IMG_20220129_112936_296 - Copy.jpg'  
    k = 5  
    
    colors, color_counts = extract_colors(image_path, k)
    print("Dominant Colors (RGB):", colors)
    print("Color Usage (Pixel Count):", color_counts)
    
    plot_colors(colors, color_counts)
