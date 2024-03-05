#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join
from torch import device,save, load, no_grad, any, isnan, autograd
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from create_dataset import ContrasiveDataset
from models import PredictorSmall

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_integer('batch_size', default = 512, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')
  flags.DEFINE_integer('epochs', default = 600, help = 'epochs to train')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device')
  flags.DEFINE_enum('dist', default = 'euc', enum_values = {'euc',}, help = 'distance type')

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  dataset = ContrasiveDataset(FLAGS.dataset, batch_size = FLAGS.batch_size)
  model = PredictorSmall()
  model.to(device(FLAGS.device))
  ce = CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
    torch.set_rng_state(ckpt['seed'])
  for epoch in range(start_epoch, FLAGS.epochs):
    model.train()
    for step, rho in enumerate(dataset):
      optimizer.zero_grad()
      rho = rho.to(device(FLAGS.device))
      fv = model(rho) # fv.shape = (batch, channel)
      fi = fv[:1] # fi.shape = (1,channel)
      fj = fv[1:2] # fj.shape = (1,channel)
      fk = fv[2:] # fk.shape = (batch - 2, channel)
      if FLAGS.dist == 'euc':
        positive = torch.sum((fi - fj) ** 2, dim = -1) # positive.shape = (batch)
        negatives = torch.sum((fi - fk) ** 2, dim = -1) # negatives.shape = (batch - 2)
      else:
        raise Exception('unknown distance method')
      logits = torch.cat([positive, negatives], dim = 0) # logits.shape = (batch - 1)
      logits = torch.unsqueeze(logits, dim = 0) # logits.shape = (1, batch - 1)
      preds = torch.softmax(logits, dim = -1) # preds.shape = (1, batch - 1)
      if any(isnan(preds)):
        print('there is nan in prediction results!')
        continue
      labels = torch.zeros((1,1)).to(torch.int32)
      loss = ce(preds, labels)
      if any(isnan(loss)):
        print('there is nan in loss!')
        continue
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(dataset) + step
      if global_steps % 100 == 0:
        print('Step #%d Epoch #%d loss %f lr %f' % (global_steps, epoch, loss, scheduler.get_last_lr()[0]))
        tb_writer.add_scalar('loss', loss, global_steps)
      if global_steps % FLAGS.save_freq == 0:
        ckpt = {'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
                'seed': torch.get_rng_state()}
        save(ckpt, join(FLAGS.ckpt, 'model.pth'))
    scheduler.step()

if __name__ == "__main__":
  add_options()
  app.run(main)

