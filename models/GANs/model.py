import torch
from pl_bolts.models.gans import DCGAN

class ConstrastDCGAN(DCGAN):
    def __init__(
        self,
        latent_dim:int,
        noise_dim: int,
        beta1: float=0.5 , 
        feature_maps_gen: int=64, 
        feature_maps_disc: int=64, 
        image_channels: int=3, 
        learning_rate: float=0.5, 
        **kwargs) -> None:
        
        super().__init__(
            beta1=beta1, 
            feature_maps_gen=feature_maps_gen, 
            feature_maps_disc=feature_maps_disc, 
            image_channels=image_channels, 
            latent_dim=latent_dim+noise_dim, 
            learning_rate=learning_rate, 
            **kwargs)
        self.noise_dim = noise_dim
        self.contrast_batch = None  

    def forward(self, noise):
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    
    def _get_fake_pred(self, real: torch.Tensor) -> torch.Tensor:
        batch_size = len(real)
        
        noise = self._get_noise(batch_size, self.noise_dim)
        noise = torch.cat((self.contrast_batch, noise), 1)

        fake = self(noise)
        fake_pred = self.discriminator(fake)
        return fake_pred