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



    def forward(self, noise):
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    
    def _get_fake_pred(self, batch: torch.Tensor) -> torch.Tensor:
        print(batch.shape)
        noise = self._get_noise(len(batch), self.noise_dim)
        noise = torch.cat((batch.flatten(1), noise), 1)

        fake = self(noise)
        fake_pred = self.discriminator(fake)
        return fake_pred

    
    def _get_disc_loss(self, real, fake):
        # Train with real
        real_pred = self.discriminator(real)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._get_fake_pred(fake)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    
    def _disc_step(self, real, fake):
        disc_loss = self._get_disc_loss(real, fake)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss