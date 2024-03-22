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
  flags.DEFINE_integer('topk', default = 1, help = 'number of top k')
  flags.DEFINE_string('output', default = 'plot.png', help = 'path to output picture')
  flags.DEFINE_enum('dist', default = 'l2', enum_values = {'l2', 'cos'}, help = 'distance to use')

def main(unused_argv):
  # 1) trainset
  trainset = np.load(FLAGS.trainset)
  res = faiss.StandardGpuResources()
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.device = 0
  if FLAGS.dist == 'l2':
    index = faiss.GpuIndexFlatL2(res, 256, flat_config)
  elif FLAGS.dist == 'cos':
    index = faiss.GpuIndexFlatIP(res, 256, flat_config)
    faiss.normalize_L2(trainset)
  index.add(trainset)
  index_cpu = faiss.index_gpu_to_cpu(index)
  faiss.write_index(index_cpu, 'faiss.index')
  # 2) evalset
  evalset = np.load(FLAGS.evalset)
  faiss.normalize_L2(evalset)
  # 3) 1-nn search
  D, I = index.search(evalset, FLAGS.topk) # D.shape = (1148800,1) I.shape = (1148800,1)
  # 3) plot
  train_labels = np.load(FLAGS.trainlabel) # train_labels.shape = (1542160)
  eval_labels = np.load(FLAGS.evallabel) # eval_labels.shape = (1148800)
  true_values = eval_labels
  if FLAGS.dist == 'l2':
    weights = np.exp(-D) / np.sum(np.exp(-D), axis = -1, keepdims = True) # weights.shape = (query num, 5)
  elif FLAGS.dist == 'cos':
    weights = np.exp(D) / np.sum(np.exp(D), axis = -1, keepdims = True) # weights.shape = (query num, 5)
  pred_values = train_labels[I] # pred_values.shape = (query_num, 5)
  pred_values = np.sum(weights * pred_values, axis = -1) # pred_values.shape = (query_num)
  dist_values = np.log10(D[:,0])

  fig, ax1 = plt.subplots()
  accurated_pred_samples = np.abs(pred_values - true_values) <= 0.01 # good_pred.shape = (query_num)
  np.savez('results.npz', y = accurated_pred_samples, x = D[:,0])
  accurated_pred_dists = D[accurated_pred_samples] # good_dists.shape = (query_num, 1)
  threshold = np.max(accurated_pred_dists)
  bad_pred_samples = D[:,0] > threshold # bad_pred.shape = (query_num)
  good_pred_samples = np.logical_and(D[:,0] <= threshold, accurated_pred_samples) # good_pred.shape = (query_num)
  hard_pred_samples = np.logical_and(D[:,0] <= threshold, np.logical_not(accurated_pred_samples))
  # draw vxc
  bad_count = np.sum(bad_pred_samples.astype(np.int32))
  good_count = np.sum(good_pred_samples.astype(np.int32))
  hard_count = np.sum(hard_pred_samples.astype(np.int32))
  print('bad_count: %d good_count: %d hard count: %d' % (bad_count, good_count, hard_count))
  ax1.scatter(true_values[hard_pred_samples], pred_values[hard_pred_samples], c = 'y', s = 2, alpha = 0.7, label = 'vxc diff > 1e-2 & extractor(rho) diff <= thres')
  ax1.scatter(true_values[bad_pred_samples], pred_values[bad_pred_samples], c = 'r', s = 2, alpha = 0.7, label = 'vxc diff > 1e-2 & extractor(rho) diff > thres')
  ax1.scatter(true_values[good_pred_samples], pred_values[good_pred_samples], c = 'b', s = 2, alpha = 0.7, label = 'vxc diff < 1e-2 & extractor(rho) diff <= thres')
  ax1.set_xlabel('true vxc')
  ax1.set_ylabel('predict vxc')
  ax1.set_ylim(-6, 1)
  ax1.legend()
  # draw rho difference
  ax2 = ax1.twinx()
  ax2.scatter(true_values, dist_values, c = 'g', s = 2, alpha = 0.7, label = 'extractor(rho) diff')
  ax2.axhline(y=np.log10(threshold), color = 'k', linestyle = '-', alpha = 0.2, label = 'rho diff = %f' % threshold)
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
