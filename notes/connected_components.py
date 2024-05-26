import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label
import matplotlib.pyplot as plt

def load_image(path):
    print("Loading image...")
    return Image.open(path)

def find_components_and_rank(image):
    print("Processing image...")
    data = np.array(image)
    height, width = data.shape[:2]
    components = np.zeros((height, width), dtype=int)
    
    # Assuming the image is in RGBA or RGB
    if data.shape[2] == 4:  # RGBA
        data = data[:, :, :3]  # Drop alpha for simplicity
    
    # Find connected components for each unique color
    unique_colors = np.unique(data.reshape(-1, data.shape[2]), axis=0)
    print(f"Found {len(unique_colors)} unique colors.")
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    component_sizes = []
    
    for color_index, color in enumerate(unique_colors):
        print(f"Processing color {color_index + 1}/{len(unique_colors)}...")
        # Create a binary mask where this color exists
        mask = np.all(data == color, axis=-1)
        labeled, num_features = label(mask, structure)
        component_sizes.extend([(np.sum(labeled == i), color_index + 1, i) for i in range(1, num_features + 1)])
        components += (labeled + np.max(components)) * mask

    # Sort components by size in descending order
    component_sizes.sort(reverse=True, key=lambda x: x[0])
    print("Sorting components by size...")

    # Create a rank dictionary
    rank_dict = {comp[1:]: idx + 1 for idx, comp in enumerate(component_sizes)}
    
    return components, rank_dict

def label_components(image, components, rank_dict):
    print("Labeling components on the image...")
    # Draw on image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    # Iterate over each component
    for (color_index, component_id), rank in rank_dict.items():
        print(f"Processing color {color_index + 1}")
        # Find the centroid of the component to place the text
        indices = np.argwhere(components == component_id + np.max(components) * (color_index - 1))
        centroid = indices.mean(axis=0).astype(int)
        position = (centroid[1], centroid[0])  # Position needs to be in (x, y)
        draw.text(position, str(rank), fill='red', font=font)
    
    return image

def main():
    image_path = 'screenshot.png'  # Adjust the path as needed
    image = load_image(image_path)
    components, rank_dict = find_components_and_rank(image)
    labeled_image = label_components(image, components, rank_dict)
    labeled_image.save('labeled_screenshot.png')
    print("Labeled image saved as 'labeled_screenshot.png'.")
    labeled_image.show()

if __name__ == "__main__":
    main()
