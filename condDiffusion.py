"""
Study of the paper: https://arxiv.org/pdf/2105.05233.pdf

https://github.com/tcapelle/Diffusion-Models-pytorch
https://github.com/dome272/Diffusion-Models-pytorch/tree/main
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

Denoising Diffusion Probabilistic Models (DDPM) implementation in PyTorch
Loads the FashionMNIST dataset
Trains a DDPM model on the dataset
Saves a sample (n images) of the model's output after each epoch
Saves a checkpoint of the model after each epoch
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl

from models.Unet import *
from utils import *
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint


class Diffusion(pl.LightningModule):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256,img_channels=1, lr=1e-4):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.loss = nn.MSELoss()
        self.lr = lr

        self.flag = False
        self.model = UNet(in_channels=img_channels, out_channels=img_channels, time_dim=256)

    def prepare_noise_schedule(self):
        #linear noise schedule
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps,device=self.device)
        return betas

    def noise_images(self, x_0, t):
        # Input: x_0 shape = (batch_size, 1, img_size, img_size)
        z = torch.randn_like(x_0) # Std normal gaussain noise with same shape as x_0
        x_t = ( torch.sqrt(self.alpha_hat[t , None, None, None]) *
                        x_0 + torch.sqrt(1.0 - self.alpha_hat[t , None, None, None]) * z ) #Noisy image
        return x_t, z
    
    def sample(self, model, num_of_samples):
        # Input: model, num_of_samples
        # Output: samples of shape (num_of_samples, 1, img_size, img_size)
        with torch.no_grad():
            x = torch.randn((num_of_samples, 1, self.img_size, self.img_size), device=self.device)
            label = torch.randint(low=0, high=10, size=(num_of_samples,),device=self.device) 
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                t = ( torch.ones(num_of_samples, device= x.device) * i).long() # step goes backwards for each image in batch

                predicted_noise = model(x, t, label)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # Sample noise for reparametrization trick
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) # No noise

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            # x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x

    #   ============= PL specific ===========
    def onepass(self, batch, batch_idx, mode):
        if not self.flag:
            self.beta = self.prepare_noise_schedule()
            self.alpha = 1. - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0)
            self.flag = True
        images, labels  = batch
        batch_size = images.shape[0]


        t = torch.randint(low=1, high=self.noise_steps, size=(batch_size,),device=self.device)
        x_t, noise = self.noise_images(images, t)

        if mode == "Val":
            with torch.no_grad():
                predicted_noise = self.model(x_t, t, labels)
        else:
            predicted_noise = self.model(x_t, t, labels)

        loss = self.loss(noise, predicted_noise)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx,mode='Train')
        self.log("Loss/Train_loss",loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.onepass(batch, batch_idx,mode='Val')
        self.log("Loss/Val_loss",loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5) # patience in the unit of epoch
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Loss/Val_loss",
                "frequency": 1
            },
        }
    
    def on_train_epoch_end(self):
        sampled_images = self.sample(self.model, num_of_samples=10)
        if not os.path.exists("./results"):
        # Create the folder
          os.makedirs("./results")
        save_images(sampled_images, os.path.join("results",  f"{self.current_epoch}.jpg"))
    
def main(n_epochs=100, AMP=True, batch_size=16):
    # ===========data===========
    training_data, test_data = get_data()
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    data_sample = next(iter(train_dataloader))
    img_size = data_sample[0].shape[-1]

    # ===========model===========
    model = Diffusion(img_size=img_size)
    
    # -----PL configs-----
    tensorboard = pl_loggers.TensorBoardLogger(save_dir="Logs/TrainLogs",name='',flush_secs=1)
    early_stop_callback = EarlyStopping(monitor='lr', stopping_threshold=2e-6, patience=n_epochs)   
    checkpoint_callback = ModelCheckpoint(filename="{epoch}",     # Checkpoint filename format
                                          save_top_k=-1,          # Save all checkpoints
                                          every_n_epochs=1,               # Save every epoch
                                          save_on_train_epoch_end=True,
                                          verbose=True)

    # train model
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=("16-mixed" if AMP else 32), max_epochs=n_epochs, 
                         callbacks=[early_stop_callback, checkpoint_callback],
                         logger=tensorboard, profiler="simple", val_check_interval=0.25, 
                         accumulate_grad_batches=1, gradient_clip_val=0.5)

    
    trainer.validate(model=model, dataloaders=test_dataloader)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    # trainer.validate(model=model, dataloaders=test_data_loader)


if __name__ == "__main__":
    main()



 
