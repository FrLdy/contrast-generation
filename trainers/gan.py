import torch
from utils.decorators import SaveLoss

class GanTrainer():
    REAL_LABEL = 1
    FAKE_LABEL = 0

    def __init__(self, 
        generator,
        discriminator, 
        n_epochs,
        dataloader,
        criterion,
        discriminator_optimizer,
        generator_optimizer
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = criterion
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.n_epochs = n_epochs
        self.dataloader = dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_latent_space = None

    def train(self):
        for epoch in range(self.n_epochs):
            for i, batch in enumerate(self.dataloader, 0):
                batch = batch[0].to(self.device)
                batch_size = batch.size(0)
                self.discriminator_training_step(batch, batch_size)
                self.generator_training_step(batch, batch_size) 


    @SaveLoss
    def discriminator_training_step(self, batch, batch_size):
        def on_real_data():
            self.discriminator.zero_grad()
            labels = self.gen_labels(batch_size, self.REAL_LABEL) 
            pred = self.discriminator(batch)
            loss = self.criterion(pred, labels)
            loss.backward()
            return loss
        
        def on_fake_data():
            fake = self.generator(self.input_latent_space)
            labels = self.gen_labels(batch_size, self.FAKE_LABEL)
            pred = self.discriminator(fake)
            loss = self.criterion(pred, labels)
            loss.backward()
            return loss

        loss = on_real_data() + on_fake_data()
        self.discriminator_optimizer.step()
        return loss

    @SaveLoss
    def generator_training_step(self, batch, batch_size):
        self.generator.zero_grad()
        labels = self.gen_labels(batch_size, self.REAL_LABEL) 
        pred = self.discriminator(batch)
        loss = self.criterion(pred, labels)
        loss.backward()
        self.generator_optimizer.step()
        return loss


    def gen_labels(self, batch_size, label):
        return torch.full((batch_size,), label, dtype=torch.float, device=self.device)

