from torch import nn
import copy


class MarioNetwork(nn.Module):
    """
    input: (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 
    """
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        #checking size of images equals 84
        c, w, h = input_dim

        if h != 84:
            raise ValueError(f'Expecting input height of 84: got {h}')
        if w != 84:
            raise ValueError(f'Expecting input width of 84: got {w}')

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)