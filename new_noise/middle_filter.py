import torch.nn as nn
from kornia.filters import MedianBlur


class MF(nn.Module):

	def __init__(self, kernel):
		super(MF, self).__init__()
		self.middle_filter = MedianBlur((kernel, kernel))

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		return self.middle_filter(image)

import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Specify the image path (modify this to your actual image path)
    image_path = "/mnt/e/dataset/quick_small/000000000285.jpg"  # Assumes the image is in the current working directory

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        exit(1)

    # Read the image and convert it to a tensor
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

    noise_layer = MF(kernel=3)


    # Process the image
    processed_jpeg = noise_layer([image_tensor,image_tensor])


    # Print the dimensions
    print("Original image size:", image_tensor.shape)
    print("After noise processing:", processed_jpeg.shape)

    # Visualize the images
    original_np = torch.clamp(image_tensor.squeeze(0), 0, 1).permute(1, 2, 0).cpu().numpy()
    processed_np = torch.clamp(processed_jpeg.squeeze(0), 0, 1).permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(processed_np)
    axes[1].set_title('After noise attack')
    axes[1].axis('off')
    plt.show()