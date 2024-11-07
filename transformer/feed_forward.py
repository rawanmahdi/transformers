import torch.nn as nn

dropout = 0.2

class FeedForward(nn.Module):
    """ simple linear layer followed by non-linearity """

    def __init__(self, n_embed, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4), # scale computation by 4
            nn.ReLU(), 
            nn.Linear(n_embed * 4, n_embed), # scale residual computation by 4
            nn.Dropout(dropout)
        )

    def forward(self, x):
        output = self.net(x)
        return output

