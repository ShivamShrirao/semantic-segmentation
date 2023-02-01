import argparse

import numpy as np
import torch
import yaml
from rich.console import Console
import onnx
from onnxsim import simplify

from semseg.datasets import *
from semseg.models import *

console = Console()


class SemSeg(torch.nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        # inference device cuda or cpu
        self.device = device
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = 255 * torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(self.device)
        self.std = 255 * torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(self.device)

    @torch.inference_mode()
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.sub(self.mean)
        x = x.div_(self.std)
        x = self.model(x)
        x = x.softmax(dim=1).argmax(dim=1)
        return x.to(torch.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/lawin.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    semseg = SemSeg(cfg, cfg['DEVICE'])

    data = torch.randint(0, 255, (1, *cfg['TEST']['IMAGE_SIZE'], 3), dtype=torch.uint8, device='cuda')

    with torch.no_grad():
        svd_out = semseg(data)

    with torch.inference_mode(), torch.jit.optimized_execution(True):
        traced_script_module = torch.jit.trace(semseg, data)
        traced_script_module = torch.jit.optimize_for_inference(traced_script_module)

    with torch.no_grad():
        o = traced_script_module(data)

    np.testing.assert_allclose(o.cpu().numpy(), svd_out.cpu().numpy(), rtol=1e-02, atol=1)
    console.print(svd_out.shape, o.shape)
    console.print(o)
    file = cfg['TEST']['MODEL_PATH'].split('.')[0]
    traced_script_module.save(f"{file}.pt")

    semseg = SemSeg(cfg, "cpu")
    torch.onnx.export(
        semseg,
        data.cpu(),
        f"{file}.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={'input' : {0 : 'batch_size'},
                      'output' : {0 : 'batch_size'}}
    )
    onnx_model = onnx.load(f"{file}.onnx")
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    onnx.save(onnx_model, f"{file}.onnx")
    assert check, "Simplified ONNX model could not be validated"
