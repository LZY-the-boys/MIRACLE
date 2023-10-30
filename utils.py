import logging
import logging, sys
from accelerate import Accelerator
import sys
import os
import torch
import json
import torch.distributed as dist
from torch import nn
import copy
from datetime import datetime
import traceback
import socket
import pandas as pd
import numpy as np
import yaml
import random
import sys
import numpy as np

DEBUG=os.environ.get('DEBUG',False)

def get_time():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def mean(data):
    if isinstance(data, dict):
        return np.array(list(data.values())).mean()
    elif isinstance(data, list):
        return np.array(data).mean()

def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('8.8.8.8', 80))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP

def word_cnt(sent):
    return len(sent.split())

def mean(data, weight=None):
    # {a:x,b:y,c:z} -> (x+y+z) /3
    if isinstance(data, dict):
        return np.average(np.array(list(data.values())),weights=weight)
    elif isinstance(data, list):
        return np.average(np.array(data),weights=weight)
    else:
        raise Exception('no implementation')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_transformers_memory(model,unit='gb' ):
    # NOTE only for huggingface transformers
    divisor = 1
    if unit.lower() == 'kb':
        divisor = 1024
    elif unit.lower() == 'mb':
        divisor = 1024*1024
    elif unit.lower() == 'gb':
        divisor = 1024*1024*1024
    else:
        raise ValueError()
    return model.get_memory_footprint() / divisor

def get_memory(info_name='memory_allocated', unit='G'):
    if info_name == 'memory_allocated':
        current_value = torch.cuda.memory.memory_allocated()
    elif info_name == 'max_memory_allocated':
        current_value = torch.cuda.memory.max_memory_allocated()
    elif info_name == 'memory_reserved':
        current_value = torch.cuda.memory.memory_reserved()
    elif info_name == 'max_memory_reserved':
        current_value = torch.cuda.memory.max_memory_reserved()
    else:
        raise ValueError()

    divisor = 1
    if unit== 'K':
        divisor = 1024
    elif unit == 'M':
        divisor = 1024*1024
    elif unit == 'G':
        divisor = 1024*1024*1024
    else:
        raise ValueError()

    # 可以通过计算差值获取模型大小
    # diff_value = current_value - self._cache_[info_name]
    # self._cache_[info_name] = current_value

    return current_value/divisor #, diff_value/divisor


def unwrap_torch_compile(state_dict):
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k.replace('_orig_mod.','')] = state_dict[k]
    del state_dict
    return new_state_dict

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

accelerator =None
def get_local_rank2():
    # return accelerator.local_process_index
    global accelerator
    if accelerator is None:
        accelerator = Accelerator()
    return accelerator.local_process_index
    # return 'cpu'
    # return 'cpu' # for debug

def gpu_clean():
    import gc;
    gc.collect()
    torch.cuda.empty_cache()

MODEL_DIR= os.environ.get('MODEL_DIR', 'model')
RANK=int(os.environ.get('LOCAL_RANK', 0))

class HyperParams:
    def from_inspect(self, args, locals):
        for n in args:
            setattr(self, n, locals[n])
        return self

    def from_dict(self, dicts):
        for n,v in dicts.items():
            setattr(self, n, v)
        return self

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}

        for name, value in sorted(self.__dict__.items()):
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

from contextlib import contextmanager
@contextmanager
def evaluating(model):
    # turn off the logger in evaluation
    state = model.training    
    try:
        model.eval()
        logging.getLogger('transformers.trainer').setLevel(logging.WARNING)
        yield model
    finally:
        if state:
            model.train()
        logging.getLogger('transformers.trainer').setLevel(logging.INFO)

class ColorFormatter(logging.Formatter):

    grey = "\x1b[30;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    purple = "\x1b[35;20m"
    light_blue = "\x1b[36;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__(fmt)
        self.FORMATS = {
            logging.DEBUG: self.grey + fmt + self.reset,
            logging.INFO: self.blue + fmt + self.reset,
            logging.WARNING: self.yellow + fmt + self.reset,
            logging.ERROR: self.red + fmt + self.reset,
            logging.CRITICAL: self.bold_red + fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def set_file_logger(name, dir, use_console=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    os.makedirs(dir, exist_ok=True)

    if use_console:
        logger.propagate = False # disable default handler
        consoleHandler = logging.StreamHandler(sys.stdout)
        if DEBUG:
            consoleHandler.setLevel(logging.DEBUG if RANK in [-1,0] else logging.WARNING )
        else:
            consoleHandler.setLevel(logging.INFO if RANK in [-1,0] else logging.WARNING )
        consoleHandler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s %(message)s"))
        logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(os.path.join(dir,'session.log'), mode='a') 
    fileHandler.setLevel(logging.INFO if RANK in [-1,0] else logging.WARNING)
    fileHandler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s %(message)s"))
    logger.addHandler(fileHandler)
    return logger

def to_jsonl(data, path, mode='w'):
    if not isinstance(data, list):
        data = [data]
    with open(path, mode) as f:
        for line in data:
            f.write(json.dumps(line,ensure_ascii=False)+'\n')

def from_jsonc(path):
    # support for json with comment 
    import jstyleson
    return jstyleson.load(open(path))

def from_json(path):
    return json.load(open(path))

def from_jsonl(path):
    return [json.loads(line) for line in open(path, 'r',encoding='utf8') ]

def to_yaml(data,path):
    yaml.safe_dump(data,open(path,'w'))

def from_yaml(path):
    return yaml.safe_load(open(path))

def to_json(data, path, mode='w'):
    if mode == 'a' and os.path.exists(path):
        old_data = from_json(path)
        data = old_data + data
    json.dump(data, open(path, 'w', encoding='utf8'), ensure_ascii=False)

# next(iter(data.items()))[1].keys()
def to_excel(data, path, index=None, columns=None, mode='w'):

    if columns is None:
        # text_df(index, 'b')
        # NOTE : { 'a':{'x''y'},'b':{'x''y'}} => rows: x,y columns: a,b
        df = pd.DataFrame(data,index=index).T
        if mode == 'a':
            if os.path.exists(path):
                previous = pd.read_excel(path,index_col=0)
                df = pd.concat([previous,df])
                df.to_excel(path,index=True)
                return
        df.to_excel(path,index=True)
    # given column
    elif index is None:
        df = pd.DataFrame(data,columns = columns)

    df.to_excel(path,index=False)

def from_excel(path):
    df = pd.read_excel(path).to_dict('records')
    return df

def get_trainable_numel(model, unit='b'):
    divisor = 1
    if unit== 'k':
        divisor = int(1e3)
    elif unit == 'm':
        divisor = int(1e6)
    elif unit == 'b': # billion=10yi
        divisor = int(1e9)
    else:
        raise ValueError()

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += param.numel()
    return {
        "trainable params": trainable_params/divisor,
        "all params": all_param/divisor, 
        "trainable%": 100 * trainable_params / all_param,
    }
