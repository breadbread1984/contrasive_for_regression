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
  flags.DEFINE_enum('dist', default = 'euc', enum_values = {'euc'}, help = 'distance method')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  rho = np.load(join(FLAGS.input_dir, 'alldata.npy'))
  vxc = np.load(join(FLAGS.input_dir, 'vxc_all.npy'))
  sample_list = list()
  for r,v in tqdm(zip(rho,vxc)):
    x = np.reshape(r[3:], (1,11,11,11))
    if FLAGS.dist == 'euc':
      dists = np.sqrt((vxc - np.expand_dims(v, axis = 0)) ** 2) # dists.shape = (sample_num,)
    else:
      raise Exception('unknown distance method')
    file_name = join(FLAGS.output_dir, str(uuid1()) + '.npz')
    np.savez(file_name, rho = x, dists = dists)
    sample_list.append(file_name)
  sample_list = np.array(sample_list)
  np.save(join(FLAGS.output_dir, 'sample_list.npy'), sample_list)

class ContrasiveDataset(Dataset):
  def __init__(self, dataset_dir, batch_size = 512):
    super(ContrasiveDataset, self).__init__()
    self.sample_list = np.load(join(dataset_dir, 'sample_list.npy'))
    self.dataset_dir = dataset_dir
    self.batch_size = batch_size
  def __len__(self):
    return len(self.sample_list)
  def __getitem__(self, idx):
    samples = list()
    # sample i
    datai = np.load(self.sample_list[idx])
    rhoi = datai['rho']
    distsi = datai['dists']
    samples.append(rhoi)
    # sample j
    while True:
      j = np.random.randint(low = 0, high = len(self.sample_list))
      dist = distsi[j]
      mask = distsi > dist
      if j != idx and np.any(mask): break
    dataj = np.load(self.sample_list[j])
    rhoj = dataj['rho']
    samples.append(rhoj)
    # sample k
    sample_num = max(self.batch_size - 2, np.sum(mask.astype(np.int32)))
    ks = np.choice(self.sample_list[mask], size = sample_num, replace = False)
    for k in ks:
      datak = np.load(k)
      rhok = datak['rho']
      samples.append(rhok)
    samples = torch.from_numpy(np.stack(samples, axis = 0))
    return samples

if __name__ == "__main__":
  add_options()
  app.run(main)

