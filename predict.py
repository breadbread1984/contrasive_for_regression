#!/usr/bin/python3

from os.path import join
import torch
from torch import load, device
from models import PredictorSmall

class Predict(object):
  def __init__(self, ckpt_path):
    ckpt = load(join(ckpt_path, 'model.pth'))
    self.model = PredictorSmall().to(torch.device('cuda'))
    self.model.load_state_dict(ckpt['state_dict'])
    self.model.eval()
  def predict(self, inputs):
    # NOTE: inputs.shape = (batch, 1, 11, 11, 11)
    if type(inputs) is np.ndarray:
      inputs = torch.from_numpy(inputs.astype(np.float32))
    inputs = inputs.to(torch.device('cuda'))
    results = self.model(inputs).cpu().detach().numpy()
    return results
