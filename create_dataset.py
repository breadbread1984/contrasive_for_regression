#!/usr/bin/python3

from absl import flags, app
from uuid import uuid1
from tqdm import tqdm
from os import listdir, mkdir
from os.path import isdir, join, exists, splitext
from shutil import rmtree
import numpy as np
from torch.utils.data import Dataset, DataLoader

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to input directory')
  flags.DEFINE_string('output_dir', default = 'dataset_torch', help = 'path to output directory')
  flags.DEFINE_integer('pool_size', default = 16, help = 'size of multiprocess pool')
  flags.DEFINE_enum('dist', default = 'euc', enum_values = {'euc', 'inn'}, help = 'distance method')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  rho = np.load(join(FLAGS.input_dir, 'alldata.npy'))
  vxc = np.load(join(FLAGS.input_dir, 'vxc_all.npy'))
  sample_list = list()
  for r,v in tqdm(zip(rho,vxc)):
    x = np.reshape(r[3:], (1,11,11,11))
    if FLAGS.dist == 'euc':
      dists = np.sqrt(np.sum((vxc - np.expand_dims(v, axis = 0)) ** 2, axis = -1)) # dists.shape = (sample_num,)
    elif FLAGS.dist == 'inn':
      dists = np.exp(
        -np.sum(
          (vxc/np.linalg.norm(vxc, axis = -1, keepdims = True)) * \
          (np.expand_dims(v, axis = 0)/np.linalg.norm(np.expand_dims(v, axis = 0), axis = -1, keepdims = True)), 
        axis = -1)
      ) # dists.shape = (sample_num,)
    file_name = join(FLAGS.output_dir, str(uuid1()) + '.npz')
    np.savez(file_name, rho = x, dists = dists)
    sample_list.append(file_name)
  sample_list = np.array(sample_list)
  np.save(join(FLAGS.output_dir, 'sample_list.npy'), sample_list)

class ContrasiveDataset(object):
  def __init__(self, dataset_dir, batch_size = 512):
    super(ContrasiveDataset, self).__init__()
    self.sample_list = np.load(join(dataset_dir, 'sample_list.npy'))
    self.dataset_dir = dataset_dir
    self.batch_size = batch_size
  def __len__(self):
    return len(sample_list)
  def __getitem__(self, idx):
    datai = np.load(self.dataset_dir, self.sample_list[idx])

if __name__ == "__main__":
  add_options()
  app.run(main)

