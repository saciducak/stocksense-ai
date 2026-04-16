import torch
import torch.nn as nn
import torch.quantization as tq
from src.models.transformer_model import TransformerPredictor

model = TransformerPredictor(input_size=24, d_model=128, nhead=8, num_encoder_layers=1, forecast_horizon=5)
model.eval()

# Quantize model
torch.backends.quantized.engine = 'qnnpack'
qmodel = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

x = torch.randn(2, 60, 24)

# Bypassing the forward method to debug internal types
x_proj = qmodel.input_projection(x)
print("Type of x after input_proj:", type(x_proj))
x_pe = qmodel.positional_encoding(x_proj)
print("Type of x after PE:", type(x_pe))

if not isinstance(x_pe, torch.Tensor):
    print("x_pe is not a tensor!")
