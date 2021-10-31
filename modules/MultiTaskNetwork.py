from modules import erfnet
import torch

class MultiTaskNetwork(torch.nn.Module):
    def __init__(self, encoder, left_decoders: dict, right_decoders: dict):
        super().__init__()
        self.encoder = encoder
        self.left_decoders = torch.nn.ModuleDict(left_decoders)
        self.right_decoders = torch.nn.ModuleDict(right_decoders)

    def forward(self, left, right=None):
        left_features = self.encoder(left)
        if len(self.right_decoders) > 0:
            right_features = self.encoder(right)

        outs = {}
        for k,v in self.left_decoders.items():
            outs[k] = v(left_features)
        if right is not None:
            for k,v in self.right_decoders.items():
                if k in outs:
                    x_right = v(right_features)
                    # concat along the channel axis
                    x_left = outs[k]
                    if isinstance(x_left, list) and isinstance(x_right, list):
                        assert len(x_left) == len(x_right)
                        outs[k] = [torch.cat([xl,xr], dim=1) for xl,xr in zip(x_left, x_right)]
                    else:
                        outs[k] = torch.cat([x_left, x_right], dim=1)
        return outs
