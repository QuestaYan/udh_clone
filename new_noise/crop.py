import torch
import torch.nn as nn
import numpy as np

def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
	image_height = image_shape[2]
	image_width = image_shape[3]

	remaining_height = int(height_ratio * image_height)
	remaining_width = int(width_ratio * image_width)

	if remaining_height == image_height:
		height_start = 0
	else:
		height_start = np.random.randint(0, image_height - remaining_height)

	if remaining_width == image_width:
		width_start = 0
	else:
		width_start = np.random.randint(0, image_width - remaining_width)

	return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class Crop(nn.Module):

	def __init__(self, height_ratio, width_ratio):
		super(Crop, self).__init__()
		self.height_ratio = height_ratio
		self.width_ratio = width_ratio

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		mask = torch.zeros_like(image)
		mask[:, :, h_start: h_end, w_start: w_end] = 1

		return image * mask

class Cropout(nn.Module):

	def __init__(self, height_ratio, width_ratio):
		super(Cropout, self).__init__()
		self.height_ratio = height_ratio
		self.width_ratio = width_ratio

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
		output = cover_image.clone()
		output[:, :, h_start: h_end, w_start: w_end] = image[:, :, h_start: h_end, w_start: w_end]
		return output

class Dropout(nn.Module):

	def __init__(self, prob):
		super(Dropout, self).__init__()
		self.prob = prob

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		rdn = torch.rand(image.shape).to(image.device)
		output = torch.where(rdn > self.prob * 1., cover_image, image)
		return output


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



    # Create instances of Jpeg and JpegTest
    # noise_layer = Crop(0.9,0.9) 
    # noise_layer = Cropout(0.5,0.6)  
    noise_layer = Dropout(0.5)


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