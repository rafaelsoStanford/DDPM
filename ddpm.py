import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from models.Unet import *
from tqdm import tqdm

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def visualizeData(data):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def get_data():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data


class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
            betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
            return betas
    
    def sample_timesteps(self, batch_size):
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,))

    def noise_images(self, x_0, t):
        # Input: x_0 shape = (batch_size, 1, img_size, img_size)
        x_0 = x_0.to(self.device)   

        z = torch.randn_like(x_0) # Std normal gaussain noise with same shape as x_0
        
        x_t = ( torch.sqrt(self.alpha_hat[t , None, None, None]) *
                        x_0 + torch.sqrt(1.0 - self.alpha_hat[t , None, None, None]) * z ) #Noisy image
        
        return x_t, z
    
    def sample(self, model, num_of_samples):
        # Input: model, num_of_samples
        # Output: samples of shape (num_of_samples, 1, img_size, img_size)
        with torch.no_grad():
            x = torch.randn((num_of_samples, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                t = ( torch.ones(num_of_samples) * i ).long().to(self.device) # step goes backwards for each image in batch

                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Sample noise for reparametrization trick
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) # No noise

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x
        
    
def train(dataloader, epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    diffusion = Diffusion(img_size=28, device=device)

    model = UNet(in_channels=1, out_channels=1, time_dim=256, device=device).to(device)

    mse = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.shape[0]

            t = diffusion.sample_timesteps(batch_size).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        # Generate n images by diffusing rand noise
        sampled_images = diffusion.sample(model, num_of_samples=10)
        if not os.path.exists("./results"):
        # Create the folder
          os.makedirs("./results")
        if not os.path.exists("./models"):
        # Create the folder
          os.makedirs("./models")
        save_images(sampled_images, os.path.join("results",  f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", f"ckpt.pt"))


#MAIN:
batch_size = 128
training_data, test_data = get_data()
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

train(train_dataloader)



 