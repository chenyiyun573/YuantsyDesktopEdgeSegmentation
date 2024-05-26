import numpy as np
import cv2
from PIL import ImageGrab, Image
import matplotlib.pyplot as plt

def capture_screen():
    # Capture the entire screen
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    return screenshot

def convert_to_edge_map(image):
    # Convert the image to a numpy array
    image_np = np.array(image)
    
    # Initialize an empty array to store the edge information
    edge_map = np.zeros(image_np.shape[:2], dtype=np.uint8)

    # Apply Sobel edge detection on each color channel
    for i in range(3):  # Loop through each color channel
        sobelx = cv2.Sobel(image_np[:,:,i], cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image_np[:,:,i], cv2.CV_64F, 0, 1, ksize=5)
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        sobel = np.hypot(abs_sobelx, abs_sobely)
        max_sobel = np.max(sobel)
        if max_sobel > 0:
            sobel = (sobel / max_sobel * 255).astype(np.uint8)
        edge_map = np.maximum(edge_map, sobel)

    # Convert the combined edge map to binary (1 for edge, 0 for non-edge)
    _, binary_edge_map = cv2.threshold(edge_map, 50, 1, cv2.THRESH_BINARY)

    return binary_edge_map

def save_image(data, filename):
    # Use plt to save the binary image
    plt.imsave(filename, data, cmap='gray')

def main():
    # Step 1: Capture the screen
    screenshot = capture_screen()
    
    # Step 2: Convert screenshot to edge map
    edge_map = convert_to_edge_map(screenshot)
    
    # Step 3: Save the edge map
    save_image(edge_map, 'edge_map.png')
    print("Edge map saved as 'edge_map.png'")

if __name__ == "__main__":
    main()
