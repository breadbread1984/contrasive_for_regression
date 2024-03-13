#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import join, exists
from shutil import rmtree
from tqdm import tqdm
import numpy as np
import torch
from torch import load, device
from models import PredictorSmall

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input', default = None, help = 'path to dataset npy')
  flags.DEFINE_string('ckpt', default = None, help = 'path to checkpoint')
  flags.DEFINE_string('output', default = 'features.npy', help = 'directory to results npy')
  flags.DEFINE_integer('batch', default = 512, help = 'batch size')

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
  predictor = Predict(FLAGS.ckpt)
  rho = np.load(FLAGS.input)
  results = list()
  step_num = np.ceil(rho.shape[0] / FLAGS.batch).astype(np.int32)
  for i in tqdm(range(step_num)):
    r = rho[i * FLAGS.batch:(i + 1) * FLAGS.batch, 3:] # r.shape = (batch, 1331)
    inputs = np.reshape(r, (r.shape[0],1,11,11,11)).astype(np.float32)
    outputs = predictor.predict(inputs)
    results.append(outputs)
  results = np.concatenate(results, axis = 0) # results.shape = (data num, 1331)
  np.save(FLAGS.output, results)

if __name__ == "__main__":
  add_options()
  app.run(main)

