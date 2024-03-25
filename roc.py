#!/usr/bin/python3

from absl import flags, app
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dist', default = 'l2', enum_values = {'l2', 'cos'}, help = 'distance type')

def main(unused_argv):
  data = np.load('results.npz')
  x, y = data['x'], data['y']
  values = np.unique(x)
  if FLAGS.dist == 'l2':
    thresholds = np.concatenate([[np.max(values) + 1,], np.sort(values)[::-1], [0,]], axis = 0)
  elif FLAGS.dist == 'cos':
    thresholds = np.concatenate([[0,], np.sort(values)], axis = 0)
  fig, ax1 = plt.subplots()
  tprs = list()
  fprs = list()
  with open('thresholds.txt', 'w') as f:
    f.write('tpr,fpr,threshold\n')
    for idx, threshold in enumerate(tqdm(thresholds)):
      if FLAGS.dist == 'l2':
        pred = x < threshold
      elif FLAGS.dist == 'cos':
        pred = x > threshold
      else:
        raise Exception('unknown distance')
      TP = np.sum(np.logical_and(pred, y).astype(np.int32))
      TP_FN = np.sum(y.astype(np.int32)) # TP + FN
      tpr = TP / np.maximum(TP_FN, 1e-32)
      FP = np.sum(np.logical_and(pred, np.logical_not(y)).astype(np.int32))
      FP_TN = np.sum(np.logical_not(y).astype(np.int32)) # N = FP + TN
      fpr = FP / np.maximum(FP_TN, 1e-32)
      tprs.append(tpr)
      fprs.append(fpr)
      f.write('%f,%f,%f\n' % (tpr, fpr, threshold))
  ax1.plot(fprs, tprs)
  ax1.set_xlabel('false positive rate')
  ax1.set_ylabel('true postive rate')
  fig.savefig('roc.png')
  with open('roc.pkl', 'wb') as f:
    f.write(pickle.dumps(fig))

if __name__ == "__main__":
  add_options()
  app.run(main)
