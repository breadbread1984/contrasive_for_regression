#!/usr/bin/python3

from os.path import exists, join
from absl import flags, app
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import faiss
import pickle

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'path to trainset npy')
  flags.DEFINE_string('evalset', default = None, help = 'path to evalset npy')
  flags.DEFINE_string('trainlabel', default = None, help = 'path to train label npy')
  flags.DEFINE_string('evallabel', default = None, help = 'path to eval label npy')
  flags.DEFINE_integer('size', default = 11, help = 'cube size')
  flags.DEFINE_enum('dist', default = 'l2', enum_values = {'l2', 'cos'}, help = 'distance type')

def main(unused_argv):
  # create index and search
  print('create index and search')
  if FLAGS.size == 11:
    trainset = np.ascontiguousarray(np.load(FLAGS.trainset)[:,3:]).astype(np.float32)
  else:
    trainset = np.ascontiguousarray(np.load(FLAGS.trainset)).astype(np.float32)
  res = faiss.StandardGpuResources()
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.device = 0
  if FLAGS.dist == 'l2':
    index = faiss.GpuIndexFlatL2(res, FLAGS.size**3, flat_config)
  elif FLAGS.dist == 'cos':
    index = faiss.GpuIndexFlatIP(res, FLAGS.size**3, flat_config)
  faiss.normalize_L2(trainset)
  index.add(trainset)
  if FLAGS.size == 11:
    evalset = np.ascontiguousarray(np.load(FLAGS.evalset)[:,3:]).astype(np.float32)
  else:
    evalset = np.ascontiguousarray(np.load(FLAGS.evalset)).astype(np.float32)
  faiss.normalize_L2(evalset)
  D, I = index.search(evalset, 1)
  train_labels = np.load(FLAGS.trainlabel)
  eval_labels = np.load(FLAGS.evallabel)
  true_values = eval_labels
  pred_values = np.squeeze(train_labels[I], axis = -1) # pred_values.shape = (query_num, 1)
  x = D[:,0]
  y = np.abs(pred_values - true_values) <= 0.01
  # roc
  print('plotting ROC')
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
      TP = np.sum(np.logical_and(pred,y).astype(np.int32))
      TP_FN = np.sum(y.astype(np.int32)) # TP + FN
      tpr = TP / np.maximum(TP_FN, 1e-32)
      FP = np.sum(np.logical_and(pred,np.logical_not(y)).astype(np.int32))
      FP_TN = np.sum(np.logical_not(y).astype(np.int32)) # N = FP + TN
      fpr = FP / np.maximum(FP_TN, 1e-32)
      tprs.append(tpr)
      fprs.append(fpr)
      f.write('%f,%f,%f\n' % (tpr, fpr, threshold))
  ax1.plot(fprs, tprs)
  ax1.set_xlabel('false positive rate')
  ax1.set_ylabel('true positive rate')
  fig.savefig('roc.png')
  with open('roc.pkl', 'wb') as f:
    f.write(pickle.dumps(fig))

if __name__ == "__main__":
  add_options()
  app.run(main)
