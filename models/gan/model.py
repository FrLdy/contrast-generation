import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, n_input, n_fm, n_output):
        """[summary]

        Args:
            n_input ([type]): embedding input size
            n_fm ([type]):  
            n_output ([type]): [description]
        """
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(n_input, n_fm * 10, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_fm * 10),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_fm * 10, n_fm * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(n_fm * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_fm * 8, n_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fm * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( n_fm * 4, n_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fm * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( n_fm * 2, n_fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fm),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_fm, n_output, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, n_input, n_fm):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_input, n_fm, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(n_fm, n_fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(n_fm * 2, n_fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(n_fm * 4, n_fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fm * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_fm * 8, n_fm * 10, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_fm * 10),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(n_fm * 10, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
