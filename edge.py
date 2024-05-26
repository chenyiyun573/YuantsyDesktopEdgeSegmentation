import numpy as np
from PIL import ImageGrab, Image

def capture_screen():
    # Capture the entire screen
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    return screenshot

def simple_difference_edge_detection(image):
    # Convert the image to a numpy array of RGB
    image_np = np.array(image)

    # Prepare an array for the edge map
    height, width, _ = image_np.shape
    edge_map = np.zeros((height, width))

    # Loop through each pixel (except for the last row and last column)
    for y in range(height - 1):
        for x in range(width - 1):
            # Calculate the difference with the right and below pixel
            right_diff = np.sum(np.abs(image_np[y, x] - image_np[y, x + 1]))
            down_diff = np.sum(np.abs(image_np[y, x] - image_np[y + 1, x]))
            
            # Check if the difference exceeds a threshold
            if right_diff > 1 or down_diff > 1:  # Adjust this threshold to your liking
                edge_map[y, x] = 1

    return edge_map

def save_image(data, filename):
    # Use PIL to save the binary image
    img = Image.fromarray(data * 255)  # Convert binary data to a format suitable for saving
    img = img.convert("L")  # Convert to grayscale
    img.save(filename)

def main():
    # Step 1: Capture the screen
    screenshot = capture_screen()
    
    # Step 2: Convert screenshot to edge map
    edge_map = simple_difference_edge_detection(screenshot)
    
    # Step 3: Save the edge map
    save_image(edge_map, 'edge_map.png')
    print("Edge map saved as 'edge_map.png'")

if __name__ == "__main__":
    main()
