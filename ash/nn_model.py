import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A residual block, a core component of the AlphaZero neural network architecture.
    It allows for the creation of much deeper networks.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # The "skip connection"
        out = F.relu(out)
        return out


class ChessNN(nn.Module):
    """
    The neural network for ChessCC, inspired by the AlphaZero architecture.
    It takes a board state tensor and outputs a policy (move probabilities) and a value (position evaluation).
    """
    def __init__(self, num_input_channels=18, num_residual_blocks=19, num_filters=256):
        super().__init__()
        self.num_residual_blocks = num_residual_blocks

        # The initial convolutional block
        self.initial_conv = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # The body of the network, composed of residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_residual_blocks)])

        # --- Policy Head ---
        # Predicts the probability of playing each possible move.
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        # There are 4672 possible moves in chess (including underpromotions)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)

        # --- Value Head ---
        # Predicts the expected outcome of the game from this position (-1 to 1).
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Common body
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy) # No softmax, as CrossEntropyLoss will be used during training

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value