from transformers import (
    TrainerCallback,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoModelWithLMHead,
    get_polynomial_decay_schedule_with_warmup, 
    GenerationConfig,
)
import tqdm
from transformers.trainer_callback import PrinterCallback
import os
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import transformers
import random
from functools import partial
import utils
import modeling
import prompt
import middleware
import wandb
import schema
import gen_cvae

# 4. start training
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer, tokenizer,decoder_tokenizer, prompter, args,**kwargs) -> None:
        super().__init__()
        self.args = args
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.prompter = prompter
        self.logger = utils.set_file_logger('transformers.trainer', trainer.args.output_dir)
        self.epoch = 0

        if args.use_attr: # for gen_attr in training; final gen will not use it
            modeling.load_scorer(self.args, use_extern_latent=False)
            for attr in self.args.attrs:
                self.args.scorer_list[attr]['latent'] = self.trainer.model.latent_cls[attr]

        self.gen_args = utils.HyperParams().from_dict({
            **args.__dict__,
            "debug": False,
            "algorithm": "ODE",
            "attr_output_path": f'{args.output_path}/personalized'
        })
        os.makedirs(f'{args.output_path}/personalized', exist_ok=True)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        if (self.epoch) % 2 + 1 == 1:
            self.trainer._save_checkpoint(self.trainer.model, trial=None, metrics=None)
        self.epoch += 1

    def _frange_cycle_zero_linear(
        self, 
        step, 
        period, # steps per loop
        beta_min = 0., 
        beta_max = 1., 
        ratio_min=0.2, 
        ratio_max=0.2
    ):
        if step % period < ratio_min * period:
            return beta_min
        elif step % period >= ratio_max * period:
            return beta_max
        else:
            k = (beta_max - beta_min) / (( 1 - ratio_max - ratio_min ) * period)
            return k * (step % period - ratio_min * period) + beta_min

    def on_step_begin(self, args, state, control, **kwargs):
        # len_data / total_batch = num_steps
        self.trainer.model.beta = self._frange_cycle_zero_linear(self.trainer.state.global_step, 40)

    # save model的时候调用
    def on_save(self, args, state, control, model, **kwargs):
        pass
    
    # TODO: how to parallel
    def test(self, model, state):
        with utils.evaluating(model), torch.no_grad():
            data, _ = modeling.load_data(schema.dev_cvae_data_path, mode='gen')
            if self.args.use_context:
                preprocess = self.prompter.preprocess_gen_context
            else:
                preprocess = self.prompter.preprocess_gen
            data = data.map(preprocess)
            dataloader = torch.utils.data.DataLoader(
                data,
                batch_size=len(data),
                collate_fn= lambda data: dict((key, [ torch.tensor(d[key]) if 'ids' in key else d[key] for d in data]) for key in data[0]),
                shuffle=False,
            )
            if self.args.use_attr:
                gen_cvae.gen_attr2(model, self.tokenizer, self.decoder_tokenizer, dataloader, self.gen_args , self.logger, state.global_step)
            else:
                gen_cvae.gen(model, self.tokenizer, self.decoder_tokenizer, dataloader, self.gen_args , self.logger, state.global_step)

    def on_log(self, args, state, control, logs, **kwargs):
        train_metrics = {
            **self.args.extra,
            **logs,
        }
        self.logger.info(train_metrics)
        wandb.log({f'train/{n}':v for n,v in train_metrics.items()})

def train_cvae(
    args,
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    args.gradient_accumulation_steps = args.total_batch // args.micro_batch
    if ddp:
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
    logger = utils.set_file_logger(__name__, args.output_path, use_console=True)
    args.ip = utils.extract_ip()
    args.time= utils.get_time()
    logger.warning(f'>>> output in  {args.output_path}')
    logger.info(f'>>> using {args}')
    
    # 0. prepare data-processing tool
    tokenizer, decoder_tokenizer = modeling.load_tokenizer(args)
    logger.info(f'>>> tokenizer {tokenizer.sep_token, tokenizer.cls_token, tokenizer.pad_token}')
    logger.info(f'>>> decoder_tokenizer {decoder_tokenizer.bos_token_id, decoder_tokenizer.eos_token_id, decoder_tokenizer.pad_token_id}')
    prompter = prompt.attribute_dialogue(tokenizer, decoder_tokenizer, args.cutoff, decoder_add_eos=True, decoder_add_bos=True)
    train_data, examples = modeling.load_data(args, mode='train-cvae') 
    if args.use_context:
        preprocess = prompter.preprocess_train_context
    else:
        preprocess = prompter.preprocess_train
    ## 1.1 check example
    examples = examples.map(preprocess)
    for example in examples:
        logger.info(f'>>> prior_input_text:\n { tokenizer.decode(example["prior_input_ids"]) }')
        logger.info(f'>>> posterior_input_text:\n { tokenizer.decode(example["posterior_input_ids"]) }')
        logger.info(f'>>> tokenize input: { example["posterior_input_ids"][:10] }...{ example["posterior_input_ids"][-10:]}')
        logger.info(f'>>> labels: { decoder_tokenizer.decode([ 0 if l==-100 else l for l in example["labels"]])}')
        logger.info(f'>>> tokenize labels: { example["decoder_input_ids"] }')
    ## 1.2 process data
    num_proc = (os.cpu_count())
    train_data = train_data.shuffle().map(preprocess, num_proc=num_proc)
    # 2. prepare model
    logger.warning((
        f">>> load model from {args.encoder_path}\n"
        f">>> load model from {args.decoder_path} "
    ))
    
    model = modeling.load_model(args, decoder_tokenizer)

    logger.warning((
        f'>>> encoder memory(G): {utils.get_transformers_memory(model.prior_encoder.bert if args.share_encoder else model.prior_encoder)} \n'
        f'>>> decoder memory(G): {utils.get_transformers_memory(model.decoder)}\n'
        f'>>> parameter(B): {utils.get_trainable_numel(model)}'
    ))

    # 3. prepare trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.total_batch//args.micro_batch,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epoch,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=args.log_steps,
            logging_first_step=True, # convenient
            evaluation_strategy="no",
            save_strategy="epoch",
            save_total_limit=1, # TODO:
            # eval_steps=args.eval_steps if args.eval_data_path else None,
            # save_steps=args.save_steps,
            output_dir=args.output_path,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.use_wandb else [],
            ignore_data_skip=args.ignore_data_skip,
        ),
        data_collator=prompter.data_collator(),
    )
    if not args.use_wandb:
        wandb.init(mode='disabled')
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(CustomCallback(trainer,tokenizer,decoder_tokenizer,prompter,args))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # save args
    utils.to_json(args.__dict__, f'{args.output_path}/train_args.json')

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if args.use_wandb:
        wandb.config.update(args, allow_val_change=True)
        logger.info({'wandb-url': wandb.run.get_url()})
    # TODO: unwrap the `_orig_mod`
    torch.save(model.state_dict(),f'{args.output_path}/checkpoint-final')

def attr_collate(features):
    # 把里边的batch取出来，否则送入模型是 [1,batch,xx]
    batch = {}
    # 全部变成tensor即可
    features = features[0]
    for k, v in features.items():
        if k not in ("attr") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack(features[k])
            else:
                batch[k] = torch.tensor(features[k])
    batch['attr'] = features['attr']
    return batch

def train_cvae_attr(
    args,
):
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    args.gradient_accumulation_steps = args.total_batch // args.micro_batch
    if ddp:
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
    logger = utils.set_file_logger(__name__, args.output_path, use_console=True)
    args.ip = utils.extract_ip()
    args.time= utils.get_time()
    logger.info(f'>>> output in  {args.output_path}')
    logger.info(f'>>> using {args}')
    
    # 0. prepare data-processing tool
    tokenizer, decoder_tokenizer = modeling.load_tokenizer(args)
    logger.info(f'>>> tokenizer {tokenizer.sep_token, tokenizer.cls_token, tokenizer.pad_token}')
    logger.info(f'>>> decoder_tokenizer {decoder_tokenizer.bos_token_id, decoder_tokenizer.eos_token_id, decoder_tokenizer.pad_token_id}')
    prompter = prompt.attribute_dialogue(tokenizer, decoder_tokenizer, args.cutoff, decoder_add_eos=True, decoder_add_bos=True)
    if args.use_context:
        preprocess = prompter.preprocess_train_context
    else:
        preprocess = prompter.preprocess_train
    train_data = {'attr':[]}
    for attr in args.attrs:
        data, examples = modeling.load_attr_data(attr, mode='train-cvae') 
        ## 1.1 check example
        examples = examples.map(preprocess)
        for example in examples:
            logger.info(f'>>> prior_input_text:\n { tokenizer.decode(example["prior_input_ids"]) }')
            logger.info(f'>>> posterior_input_text:\n { tokenizer.decode(example["posterior_input_ids"]) }')
            logger.info(f'>>> tokenize input: { example["posterior_input_ids"][:10] }...{ example["posterior_input_ids"][-10:]}')
            logger.info(f'>>> labels: { decoder_tokenizer.decode([ 0 if l==-100 else l for l in example["labels"]])}')
            logger.info(f'>>> tokenize labels: { example["decoder_input_ids"] }')
        ## 1.3 append to set
        data = data.map(preprocess, num_proc=(os.cpu_count())).shuffle() # id之间要打乱
        data = data.remove_columns(['input','output','personas'])
        dataloader = torch.utils.data.DataLoader(
            data, 
            batch_size=args.micro_batch,
            collate_fn = prompter.data_collator(),
            # pin_memory=True,
            # num_workers=4,
        )
        for cnt in iter(dataloader):
            for k,v in cnt.items():
                if k not in train_data:
                    train_data[k] = []
                train_data[k].append(cnt[k])
            train_data['attr'].append(attr)
    train_data = Dataset.from_dict(train_data)
    columns = list(train_data.features.keys())
    train_data.set_format(columns=columns+['attr'])

    # 2. prepare model
    logger.info((
        f">>> load model from {args.encoder_path}\n"
        f">>> load model from {args.decoder_path} "
    ))
    if 'llama' in args.decoder_path:
        model = modeling.load_model2(args,decoder_tokenizer)
    else:
        model = modeling.load_model(args, decoder_tokenizer)

    logger.info((
        f'>>> encoder memory(G): {utils.get_transformers_memory(model.prior_encoder.bert if args.share_encoder else model.prior_encoder)} \n'
        f'>>> decoder memory(G): {utils.get_transformers_memory(model.decoder)}\n'
        f'>>> parameter(B): {utils.get_trainable_numel(model)}'
    ))

    # save args 注意放后面会存scorer-list, 不是 JSON serializable 的
    utils.to_json(args.__dict__, f'{args.output_path}/train_args.json')

    # 3. prepare trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            # per_device_train_batch_size=args.micro_batch,
            per_device_train_batch_size=1, # 里边已经分好batch了 走default_collate
            gradient_accumulation_steps=args.total_batch//args.micro_batch,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epoch,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=args.log_steps,
            logging_first_step=True, # convenient
            evaluation_strategy="no",
            save_strategy="no",
            # save_total_limit=1,
            # eval_steps=args.eval_steps if args.eval_data_path else None,
            # save_steps=args.save_steps,
            output_dir=args.output_path,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.use_wandb else [],
            ignore_data_skip=args.ignore_data_skip,
        ),
        
        data_collator= attr_collate,
    )
    if not args.use_wandb:
        wandb.init(mode='disabled')
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(CustomCallback(trainer,tokenizer,decoder_tokenizer,prompter,args))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model) # for 3090/4090 好像没啥加速效果

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if args.use_wandb:
        wandb.config.update(args, allow_val_change=True)
        logger.info({'wandb-url': wandb.run.get_url()})
    # TODO: unwrap the `_orig_mod`
    torch.save(model.state_dict(),f'{args.output_path}/checkpoint-final')

def _start(
    *, 
    task_name: str=None,
    cutoff: int=512,
    data_path: str=None,
    output_path: str=None,
    dev_data_path: str=None,
    use_wandb:bool=False, 
    CUTOFF_LEN:int=128, 
    encoder_path:str='bert-base-uncased',
    decoder_path:str='microsoft/DialoGPT-medium',
    kl_ratio:float=1.,
    share_encoder:bool=False,
    full_decoder:bool=False,
    micro_batch:int=4,
    total_batch:int=32,
    warmup_ratio:float= 0.05,
    num_epoch:int=100,
    latent_size:int=64,
    learning_rate:float=5e-5,
    log_steps:int=100,
    int8:bool=False,
    ignore_data_skip:bool=False,
    resume_from_checkpoint:str=None,
    use_context:bool=False,
    attrs: str=None, # 0,1,2
    ids: str=None, # 0,0;0,1;1,0;1,1
    attr_cls_ratio: float=0.,
    attr_gap_ratio: float=0.,
    prior_cls: bool=False,
    prior_gap: bool=False,
    negap: bool=False
):
    import inspect
    frame = inspect.currentframe()
    names, _, _, locals = inspect.getargvalues(frame)
    args = utils.HyperParams().from_inspect(names, locals)

    if modeling.parse_attr(args):
        args.use_attr=True
        train_cvae_attr(args)
    else:
        train_cvae(args)

if __name__ == "__main__":
    import defopt
    defopt.run(_start)