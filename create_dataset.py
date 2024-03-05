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

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  rho = np.load(join(FLAGS.input_dir, 'alldata.npy'))
  vxc = np.load(join(FLAGS.input_dir, 'vxc_all.npy'))
  sample_list = list()
  for r,v in tqdm(zip(rho,vxc)):
    x = np.reshape(r[3:], (1,11,11,11))
    m = vxc > v
    file_name = str(uuid1()) + '.npz'
    np.savez(file_name, rho = x, m = m)
    sample_list.append((file_name, np.sum(m.astype(np.int32))))
  sample_list = np.array(sample_list)
  np.save(join(FLAGS.output_dir, 'sample_list.npy'), sample_list)

class ContrasiveDataset(object):
  def __init__(self, dataset_dir, batch_size = 512, epoch_size = 50000):
    super(ContrasiveDataset, self).__init__()
    self.sample_list = np.load(join(dataset_dir, 'sample_list.npy'))
    self.batch_size = batch_size
    self.epoch_size = epoch_size
  def __len__(self):
    return self.epoch_size
  def __getitem__(self, idx):
    pass

if __name__ == "__main__":
  add_options()
  app.run(main)

