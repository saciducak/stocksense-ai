import torch
import torch.nn as nn
import torch.quantization as tq
from src.models.transformer_model import TransformerPredictor

model = TransformerPredictor(input_size=24, d_model=128, nhead=8, num_encoder_layers=1, forecast_horizon=5)
model.eval()

torch.backends.quantized.engine = 'qnnpack'

# Ignore all MultiheadAttention elements, but wait, MHA doesn't have nn.Linear inside it typically except out_proj!
# Just quantizing Linear causes the crash because out_proj is inside MHA.
qconfig = torch.quantization.default_dynamic_qconfig
qconfig_spec = {
    nn.Linear: qconfig,
    nn.LSTM: qconfig,
    nn.GRU: qconfig,
}
# Can we disable MHA quantization?
# MHA is not dynamically quantized by default anyway unless we quantize Linear and out_proj gets caught.
# MHA is considered a leaf module in some versions, but out_proj is nn.NonDynamicallyQuantizableLinear in newer PyTorch to PREVENT this exact bug!
# Let's see if MHA has NonDynamicallyQuantizableLinear in our PyTorch version.

# Easy workaround: just patch the transformer model to run self.transformer_encoder normally but we can't if out_proj is quantized.
