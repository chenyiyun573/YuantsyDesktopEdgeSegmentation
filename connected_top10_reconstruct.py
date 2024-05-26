import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label

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
        mask = np.all(data == color, axis=-1)
        labeled, num_features = label(mask, structure)
        component_sizes.extend([(np.sum(labeled == i), color_index, i) for i in range(1, num_features + 1)])
        components += (labeled + np.max(components)) * mask

    # Sort components by size in descending order
    component_sizes.sort(reverse=True, key=lambda x: x[0])
    print("Sorting components by size...")

    # Keep only the top 10 components
    top_component_sizes = component_sizes[:10]

    # Generate rank dict to use colors correctly
    rank_dict = {comp[1]: idx + 1 for idx, comp in enumerate(top_component_sizes)}
    
    return components, unique_colors, rank_dict

def reconstruct_image(original_image, components, unique_colors, rank_dict):
    print("Reconstructing image with top 10 components...")
    data = np.array(original_image)
    height, width, channels = data.shape
    reconstructed_image = np.zeros_like(data)

    # Adjust the background color to match the number of channels in the original image
    if channels == 4:  # RGBA
        background_color = [255, 255, 255, 255]  # white background with full opacity
    else:  # RGB
        background_color = [255, 255, 255]

    # Set the entire image to the background color
    reconstructed_image[:] = background_color

    # Reconstruct the image using only the top components
    for size, color_index, component_id in rank_dict.keys():
        color = unique_colors[color_index]
        mask = (components == component_id + np.max(components) * color_index)
        # Ensure the color is assigned correctly even if the original data includes an alpha channel
        if channels == 4 and len(color) == 3:
            color = np.append(color, 255)  # Append full opacity for the color
        reconstructed_image[mask] = color
    
    return Image.fromarray(reconstructed_image)


def main():
    image_path = 'screenshot.png'  # Adjust the path as needed
    image = load_image(image_path)
    components, unique_colors, rank_dict = find_components_and_rank(image)
    reconstructed_image = reconstruct_image(image, components, unique_colors, rank_dict)
    reconstructed_image.save('reconstructed_screenshot.png')
    print("Reconstructed image saved as 'reconstructed_screenshot.png'.")
    reconstructed_image.show()

if __name__ == "__main__":
    main()
