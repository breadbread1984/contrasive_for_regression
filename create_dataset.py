#!/usr/bin/python3

from os import listdir, mkdir
from os.path import isdir, join, exists, splitext
from shutil import rmtree
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ContrasiveDataset(Dataset):
  def __init__(self, dataset_dir, dist = 'euc', batch_size = 512):
    super(ContrasiveDataset, self).__init__()
    self.rho = np.load(join(dataset_dir, 'alldata.npy'))
    self.vxc = np.load(join(dataset_dir, 'vxc_all.npy'))
    self.dist = dist
    self.batch_size = batch_size
  def __len__(self):
    return self.rho.shape[0]
  def __getitem__(self, idx):
    samples = list()
    # sample i
    rhoi = np.reshape(self.rho[idx][3:],(1,11,11,11))
    if self.dist == 'euc':
      distsi = np.sqrt((self.vxc - self.vxc[idx:idx+1]) ** 2) # dists.shape = (sample_num,)
    elif self.dist == 'l1':
      distsi = np.abs(self.vxc - self.vxc[idx:idx+1]) # dists.shape = (sample_num,)
    samples.append(rhoi)
    # sample j
    while True:
      j = np.random.randint(low = 0, high = self.rho.shape[0])
      dist = distsi[j]
      mask = distsi > dist
      if j != idx and np.any(mask): break
    rhoj = np.reshape(self.rho[j][3:],(1,11,11,11))
    samples.append(rhoj)
    # sample k
    sample_num = max(self.batch_size - 2, np.sum(mask.astype(np.int32)))
    ks = np.choice(self.rho[mask], size = sample_num, replace = False)
    for k in ks:
      rhok = np.reshape(k[3:],(1,11,11,11))
      samples.append(rhok)
    samples = torch.from_numpy(np.stack(samples, axis = 0))
    return samples

