import cv2
import numpy as np
import os

dataset_name = "brain_tumor_mri"
set_types = ["Training", "Testing"]

def draw_shapes(img, shape, color_type):
    """
    Draw a specified shape at the center slightly above the middle of the image.
    The shape is filled with the specified color type.
    :param img: Original image
    :param shape: The shape to draw ('triangle', 'circle', 'concave_polygon')
    :param color_type: The fill color type ('white', 'gray', 'noise')
    :return: Image with the added shape
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2 - h // 8)  # Center slightly above the middle

    tumor_height = h // 8
    tumor_width = w // 8

    # Create a copy to avoid modifying the original image
    img_copy = img.copy()

    # Define fill color
    if color_type == 'white':
        fill_color = (255, 255, 255)  # White
    elif color_type == 'gray':
        fill_color = (127, 127, 127)  # Gray
    elif color_type == 'noise':
        # Noise as grayscale
        fill_color = None  # Will use grayscale noise

    def apply_shape(points):
        # Draw the shape with a black border
        cv2.polylines(img_copy, [points], isClosed=True, color=(0, 0, 0), thickness=1)
        if color_type == 'noise':
            # Fill the shape region with grayscale noise
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)  # Create a mask for the shape region
            noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)  # Generate grayscale noise
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray[mask > 0] = noise[mask > 0]
            img_copy[:, :, 0] = img_copy[:, :, 1] = img_copy[:, :, 2] = img_gray
        else:
            # Fill the shape with the specified color
            cv2.fillPoly(img_copy, [points], fill_color)

    if shape == "triangle":
        points = np.array([
            [center[0], center[1] - 10],  # Top
            [center[0] - 10, center[1] + 10],  # Bottom-left
            [center[0] + 10, center[1] + 10]   # Bottom-right
        ], np.int32).reshape((-1, 1, 2))
        apply_shape(points)

    elif shape == "circle":
        if color_type == 'noise':
            # Fill the circular region with grayscale noise
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, 10, 255, thickness=-1)
            noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)  # Generate grayscale noise
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray[mask > 0] = noise[mask > 0]
            img_copy[:, :, 0] = img_copy[:, :, 1] = img_copy[:, :, 2] = img_gray
        else:
            noise_x = np.random.randint(-tumor_width//2, tumor_width//2+1)
            noise_y = np.random.randint(-tumor_height//2, tumor_height//2+1)
            noisy_center = (center[0] + noise_x, center[1] + noise_y)
            cv2.circle(img_copy, noisy_center, (tumor_height+tumor_width)//2, fill_color, thickness=-1)

    elif shape == "concave_polygon":
        points = np.array([
            [center[0] - 10, center[1] - 5],  # Top-left
            [center[0], center[1] - 10],      # Top
            [center[0] + 10, center[1] - 5],  # Top-right
            [center[0] + 5, center[1] + 5],   # Concave point
            [center[0] + 10, center[1] + 10], # Bottom-right
            [center[0] - 10, center[1] + 10], # Bottom-left
            [center[0] - 5, center[1] + 5]    # Concave point
        ], np.int32).reshape((-1, 1, 2))
        apply_shape(points)

    return img_copy


def process_images_with_shapes(set_type, shape, color_type):
    # Create folders for each shape and color type
    # for shape in ['triangle', 'circle', 'concave_polygon']:
    #     for color_type in ['white', 'gray', 'noise']:
    #         folder = os.path.join(output_folder, f"{shape}_{color_type}_images")
    #         os.makedirs(folder, exist_ok=True)

    input_folder = os.path.join("data", f"{dataset_name}", f"{set_type}", "notumor")
    output_folder = os.path.join("data", f"{dataset_name}_fake", f"{dataset_name}_{shape}_{color_type}", f"{set_type}")

    os.makedirs(os.path.join(output_folder, "notumor"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, f"tumor_{shape}_{color_type}"), exist_ok=True)

    # Iterate over all images in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue  # Skip non-image files

        # Read the image
        img = cv2.imread(input_path)

        # Add shapes and save
        # for shape in ['triangle', 'circle', 'concave_polygon']:
        #     for color_type in ['white', 'gray', 'noise']:
                # shaped_img = draw_shapes(img, shape, color_type)
                # folder = os.path.join(output_folder, f"{shape}_{color_type}_images")
                # output_path = os.path.join(folder, f"{filename.split('.')[0]}_{shape}_{color_type}.png")
                # cv2.imwrite(output_path, shaped_img)
        
        shaped_img = draw_shapes(img, shape, color_type)
        output_path = os.path.join(output_folder, f"tumor_{shape}_{color_type}", f"{filename.split('.')[0]}_{shape}_{color_type}.png")
        cv2.imwrite(output_path, shaped_img)

        output_path = os.path.join(output_folder, "notumor", f"{filename}")
        cv2.imwrite(output_path, img)

    print(f"Image processing completed. Shapes saved to the specified folders.")


if __name__ == "__main__":

    for set_type in set_types:  # set_types = ["Training", "Testing"]
        process_images_with_shapes(set_type, "circle", "white")

