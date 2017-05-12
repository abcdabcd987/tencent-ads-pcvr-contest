import json
import os
import sys
import time

import __init__

sys.path.append(__init__.config['data_path'])
from datasets import as_dataset

from models import as_model
from trainers import Trainer

data_name = 'tencent-ads-pcvr-contest-pre'
model_name = 'lr'

dataset = as_dataset(data_name)
data_param = {
    'pos_ratio': None,
    'val_ratio': None,
    'batch_size': 2000,
    'random_sample': 7,
    'shuffle_block': False,
    'split_fields': False,
    'on_disk': True,
}

train_gen = dataset.batch_generator(gen_type='train', **data_param)
test_gen = dataset.batch_generator(gen_type='test', **data_param)
#test_gen = dataset.batch_generator(gen_type='test', **data_param)

model_param = {
    'init': 'uniform',
    'num_inputs': dataset.max_length,
    'input_dim': dataset.num_features,
    'factor': 4,
    'l2_w': 1e-6,
    'l2_v': 1e-6,
    'noisy': 0.1,
    'norm': True,
    # 'layer_sizes': None,
    # 'layer_acts': None,
    'layer_keeps': None,
    # 'batch_norm': False,
}

model = as_model(model_name, **model_param)

tag = time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time()))
logdir = '../log/' + dataset.__class__.__name__ + '/' + model.__class__.__name__ + '/' + tag

train_param = {
    'optimizer': 'adam',
    'loss': 'weight',
    'pos_weight': 1.0,
    'n_epoch': 10,
    'train_per_epoch': dataset.train_size,
    'test_per_epoch': dataset.test_size,
    'batch_size': data_param['batch_size'],
    'learning_rate': 0.001,
    'decay_rate': 0.9,
    'logdir': logdir,
    'load_ckpt': False,
    'ckpt_time': 10000,
    'layer_keeps': model_param['layer_keeps'],
    'percentile': False,
}

if not os.path.exists(logdir):
    os.makedirs(logdir)
if os.path.isfile(logdir + '/config.json'):
    tmp = json.load(open(logdir + '/config.json'))
else:
    tmp = {}
tmp[tag] = {
    'config': __init__.config,
    'data_name': data_name,
    'data_param': data_param,
    'model_name': model_name,
    'model_param': model_param,
    'train_param': train_param,
}
json.dump(tmp, open(logdir + '/config.json', 'w'), indent=4, sort_keys=True, separators=(',', ':'))

trainer = Trainer(model=model, train_gen=train_gen, test_gen=test_gen, **train_param)
trainer.fit()
