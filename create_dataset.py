#!/usr/bin/python3

from os import listdir, mkdir
from os.path import isdir, join, exists, splitext
from shutil import rmtree
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ContrasiveDataset(Dataset):
  def __init__(self, npy_path, dist = 'euc', batch_size = 512):
    super(ContrasiveDataset, self).__init__()
    self.rho = np.load(npy_path)[:,:1361]
    self.vxc = np.load(npy_path)[:,-1]
    self.dist = dist
    self.batch_size = batch_size
  def __len__(self):
    return self.rho.shape[0]
  def __getitem__(self, idx):
    samples = list()
    # sample i
    rhoi = np.reshape(self.rho[idx],(1,1361))
    if self.dist == 'euc':
      distsi = np.sqrt((self.vxc - self.vxc[idx:idx+1]) ** 2) # dists.shape = (sample_num,)
    elif self.dist == 'l1':
      distsi = np.abs(self.vxc - self.vxc[idx:idx+1]) # dists.shape = (sample_num,)
    samples.append(rhoi)
    # sample j
    while True:
      j = np.random.randint(low = 0, high = self.rho.shape[0])
      mask = distsi > distsi[j]
      if j != idx and np.any(mask): break
    rhoj = np.reshape(self.rho[j],(1,1361))
    samples.append(rhoj)
    # sample k
    sample_num = min(self.batch_size - 2, np.sum(mask.astype(np.int32)))
    subset = self.rho[mask]
    ks = np.random.choice(np.arange(subset.shape[0]), size = sample_num, replace = False)
    for k in ks:
      rhok = np.reshape(subset[k],(1,1361))
      samples.append(rhok)
    samples = np.stack(samples, axis = 0).astype(np.float32)
    return samples

