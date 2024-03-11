#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import join, exists
from shutil import rmtree
import torch
from torch import load, device
from models import PredictorSmall

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to dataset npy')
  flags.DEFINE_string('ckpt', default = None, help = 'path to checkpoint')
  flags.DEFINE_string('output', default = 'features', help = 'directory to output')

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

def main(unused_argv):
  if exists(FLAGS.output): rmtree(FLAGS.output)
  mkdir(FLAGS.output)
  predictor = Predictor(FLAGS.ckpt)
  rho = np.load(FLAGS.input)
  batch = list()
  for idx, r in enumerate(rho):
    inputs = np.reshape(r[3:], (1,1,11,11,11)).astype(np.float32)
    outputs = predictor.predict(inputs)
    np.save(join(FLAGS.output, '%d.npy' % idx), outputs)

if __name__ == "__main__":
  add_options()
  app.run(main)

