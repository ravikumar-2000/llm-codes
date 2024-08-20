import spacy
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch_utils import save_checkpoint, load_checkpoint

import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import Multi30k