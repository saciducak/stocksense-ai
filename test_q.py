import torch
import torch.nn as nn
import torch.quantization as tq
from src.models.transformer_model import TransformerPredictor

model = TransformerPredictor(input_size=24, d_model=128, nhead=8, num_encoder_layers=1, forecast_horizon=5)
model.eval()

x = torch.randn(2, 60, 24)

# Create a mock wrapper for TransformerEncoderLayer
layer = model.transformer_encoder.layers[0]
old_forward = layer.forward
def new_forward(src, *args, **kwargs):
    print("layer args:", [type(a) for a in args])
    print("layer kwargs:", {k: type(v) for k, v in kwargs.items()})
    return old_forward(src, *args, **kwargs)
layer.forward = new_forward

torch.backends.quantized.engine = 'qnnpack'
qmodel = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

try:
    out_q = qmodel(x)
except Exception as e:
    import traceback
    traceback.print_exc()
