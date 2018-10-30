import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.fsvae import FSVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=1000000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000,   help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,       help="Run ID. In case you want to run replicates")
parser.add_argument('--sample',    type=int, default=0,       help="Sample")
args = parser.parse_args()
layout = [
    ('model={:s}',  'fsvae'),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
fsvae = FSVAE(name=model_name).to(device)
writer = ut.prepare_writer(model_name, overwrite_existing=args.sample==0)

if args.sample:
    with torch.no_grad():
       ut.load_model_by_name(fsvae, global_step=args.iter_max)
       images = fsvae.sample_clipped_x(20)
       save_image(images.view(200, 3, 32, 32), 'fsvae_sample.png', nrow=20)
else:
    train_loader, labeled_subset, test_set = ut.get_svhn_data(device)
    train(model=fsvae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          y_status='fullsup',
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)

