#!/usr/bin/python3

from absl import flags, app
from os.path import join, exists
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import faiss

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'path to trainset npy')
  flags.DEFINE_string('evalset', default = None, help = 'path to evalset npy')
  flags.DEFINE_string('trainlabel', default = None, help = 'path to train labels')
  flags.DEFINE_string('evallabel', default = None, help = 'path to eval labels')
  flags.DEFINE_string('output', default = 'plot.png', help = 'path to output picture')

def main(unused_argv):
  # 1) trainset
  trainset = np.load(FLAGS.trainset)
  res = faiss.StandardGpuResources()
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.device = 0
  index = faiss.GpuIndexFlatL2(res, 256, flat_config)
  index.add(trainset)
  # 2) evalset
  evalset = np.load(FLAGS.evalset)
  # 3) 1-nn search
  D, I = index.search(evalset, 1) # D.shape = (1148800,1) I.shape = (1148800,1)
  # 3) plot
  train_labels = np.load(FLAGS.trainlabel) # train_labels.shape = (1542160)
  eval_labels = np.load(FLAGS.evallabel) # eval_labels.shape = (1148800)
  true_values = eval_labels
  pred_values = train_labels[I[:,0]]
  dist_values = np.log10(D[:,0])

  fig, ax1 = plt.subplots()
  # draw vxc
  bad_pred = np.abs(pred_values - true_values) > 0.01 # bad_pred.shape = (1148800)
  ax1.scatter(true_values[bad_pred], pred_values[bad_pred], c = 'r', s = 2, alpha = 0.7, label = 'vxc diff > 1e-2')
  ax1.scatter(true_values[np.logical_not(bad_pred)], pred_values[np.logical_not(bad_pred)], c = 'b', s = 2, alpha = 0.7, label = 'vxc diff < 1e-2')
  ax1.set_xlabel('true vxc')
  ax1.set_ylabel('predict vxc')
  ax1.set_ylim(-6, 1)
  ax1.legend()
  # draw rho difference
  ax2 = ax1.twinx()
  ax2.scatter(true_values, dist_values, c = 'g', s = 2, alpha = 0.7, label = 'extractor(rho) diff')
  ax2.axhline(y=np.log10(3.5e-3), color = 'k', linestyle = '-', alpha = 0.2, label = 'rho diff = 3.5e-3')
  ax2.set_ylabel('log10(extractor(rho) diff)')
  ax2.set_ylim(-5, 2)
  ax2.legend()
  fig.savefig(FLAGS.output)
  with open('plot.pkl', 'wb') as f:
    pickle.dump(fig, f)
  fig.show()

if __name__ == "__main__":
  add_options()
  app.run(main)

