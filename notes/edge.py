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
    
    # Convert RGB to Grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and detail in the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Canny Edge Detector
    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
    
    # Convert edges to binary map (1 for edge, 0 for non-edge)
    edge_map = (edges > 0).astype(int)
    
    return edge_map

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
