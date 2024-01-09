from PIL import Image
import os
import random

# Function to create a grid image
def create_grid_image(base_path, num_folders=6, images_per_folder=6, grid_size=(6, 6)):
    # Initialize the list to store images
    images = []

    # Iterate over each folder
    for folder_num in range(1, num_folders + 1):
        folder_path = os.path.join(base_path, str(folder_num))
        
        # Get all PNG images in the folder
        all_images = [img for img in os.listdir(folder_path) if img.endswith('.png')]
        
        # Randomly select images_per_folder images
        selected_images = random.sample(all_images, min(images_per_folder, len(all_images)))

        # Open and append images to the list
        for img_name in selected_images:
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path)
            images.append(img)

    # Determine individual image size (assuming all images are of the same size)
    img_width, img_height = images[0].size

    # Create a new blank image with the appropriate size
    grid_image = Image.new('RGB', (img_width * grid_size[0], img_height * grid_size[1]))

    # Paste images into the grid
    for i, img in enumerate(images):
        grid_x = (i % grid_size[0]) * img_width
        grid_y = (i // grid_size[0]) * img_height
        grid_image.paste(img, (grid_x, grid_y))

    return grid_image

# Example usage (this will not work here due to the lack of actual image files)
base_path = '/path/data/sampling/mask_09_60ksteps_finetune'
grid_image = create_grid_image(base_path)
grid_image.save("./output.png") # Or grid_image.save('output.png') to save the image