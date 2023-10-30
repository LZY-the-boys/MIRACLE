from transformers import (
    TrainerCallback,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoModelWithLMHead,
    get_polynomial_decay_schedule_with_warmup, 
    GenerationConfig,
)
import schema
import tqdm
from transformers.trainer_callback import PrinterCallback
import os
import sys
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
import transformers
import random
from functools import partial
import utils
import modeling
import prompt
import middleware
import wandb

# latent classification 
class LatentDataset():

    def __init__(self, latent_z, labels):
        self.latent_z = latent_z
        self.labels = labels

    def __len__(self):
        return len(self.latent_z)

    def __getitem__(self, idx):
        return self.latent_z[idx],self.labels[idx]

    def collate_fn(self,batch):
        transposed = list(zip(*batch))
        return {
            'input_ids': torch.tensor(transposed[0]),
            'labels': torch.tensor(transposed[1]),
        }

def focal_forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    from transformers import SequenceClassifierOutput
    from torchvision.ops import focal_loss
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = outputs[1]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
        loss_fct = focal_loss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def compute_metrics(pred):
    labels = torch.tensor(pred.label_ids).long()
    # NOTE here can't be float16, must be float:
    # <class 'RuntimeError'> "softmax_lastdim_kernel_impl" not implemented for 'Half'  
    preds = torch.softmax(torch.tensor(pred.predictions,dtype=float),dim=-1)
    # out[i][j] = preds[i][labels[i][j]]
    probs = torch.gather(preds, 1,labels.view(-1, 1))
    acc = torch.mean(probs).item()
    return {
        'accuracy': round(acc,6)
    }

def prepare_latent_classify_data(
    preprocess, gen_mid, data
):
    data = data.map(preprocess, num_proc=os.cpu_count())
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=len(data), # NOTE only one step here
        collate_fn= lambda da: dict((key, [ torch.tensor(d[key]) if 'ids' in key else d[key] for d in da]) for key in da[0]),
        shuffle=False, 
    )
    for _ , batch in enumerate((dataloader)):
        # NOTE input_ids need to be list[tensor], cannot pad!
        zs = gen_mid.sample_batched(
            batch["input_ids"],
            batch_size=4,
        )
        zlabel = batch['tag']
    data = LatentDataset(zs.tolist(), zlabel)
    return data

def train_latent_classify(
    args,
):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    args.gradient_accumulation_steps = args.total_batch // args.micro_batch
    if ddp:
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
    logger = utils.set_file_logger('transformers.trainer', args.output_path, use_console=True)
    args.ip = utils.extract_ip()
    args.time= utils.get_time()
    logger.warning(f'>>> task {args.task_name} ; output in  {args.output_path}')
    logger.info(f'>>> using {args}')
    
    # 0. prepare tokenizer
    tokenizer, decoder_tokenizer = modeling.load_tokenizer(args)
    # 1. load data
    prompter = prompt.attribute_dialogue(tokenizer, decoder_tokenizer, args.cutoff, decoder_add_eos=True, decoder_add_bos=True)
    train_data, examples = modeling.load_attr_data(args.attr, mode='train-latent') 
    eval_data, eval_examples = modeling.load_attr_data(args.attr, mode='dev-latent')
    if args.use_context:
        preprocess = prompter.preprocess_gen_context
    else:
        preprocess = prompter.preprocess_gen

    examples = examples.map(preprocess)
    for example in examples:
        logger.info(f'>>> prior_input_ids example:\n { tokenizer.decode(example["input_ids"]) }')
        logger.info(f'>>> tokenizer example: { example["input_ids"][:10] }...{ example["input_ids"][-10:]}')
        logger.info(f'>>> tokenizer label: { decoder_tokenizer.decode([0 if l==-100 else l for l in example["label_ids"]])}')
        logger.info(f'>>> tokenizer label_ids: { example["label_ids"][:10]}...{example["label_ids"][-10:]}')

    # 2. prepare model
    cvae = modeling.load_model(args, decoder_tokenizer)

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-final')
    logger.info(f'>>> load trained checkpoint from : { checkpoint_path }')
    state_dict = torch.load(checkpoint_path)
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k.replace('_orig_mod.','')] = state_dict[k]
    del state_dict
    cvae.load_state_dict(new_state_dict, strict=False)

    model = modeling.LinearClassifier(input_dim=args.latent_size, out_dim=2)
    logger.warning((
        f'>>> totol loading memory(G): {utils.get_memory()} \n'
        f'>>> model parameter(B): {utils.get_trainable_numel(model)}'
    ))
    # 3. prepare latent z to classify
    with utils.evaluating(cvae), torch.no_grad(): # or else will cuda OOM
        gen_mid = modeling.VAEInferMiddleWare(cvae, tokenizer, decoder_tokenizer)
        train_data = prepare_latent_classify_data(preprocess, gen_mid,train_data)
        eval_data = prepare_latent_classify_data(preprocess,gen_mid,eval_data)

    # 4. prepare trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch,
            per_device_eval_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.total_batch//args.micro_batch,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epoch,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_strategy="epoch",
            logging_first_step=True, # convenient
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=2, # save best / last
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            output_dir=args.output_path,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.use_wandb else [],
            ignore_data_skip=args.ignore_data_skip,
        ),
        data_collator=train_data.collate_fn,
        compute_metrics=compute_metrics,
    )
    if not args.use_wandb:
        wandb.init(mode='disabled')
    trainer.remove_callback(PrinterCallback)

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # save args
    utils.to_json(args.__dict__, f'{args.output_path}/train_args.json')

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    torch.save(utils.unwrap_torch_compile(model.state_dict()),f'{args.output_path}/best')
    if args.use_wandb:
        wandb.config.update(args, allow_val_change=True)
        logger.info({'wandb-url': wandb.run.get_url()})

def train_nli_classify(
    args,
):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    args.gradient_accumulation_steps = args.total_batch // args.micro_batch
    if ddp:
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
    logger = utils.set_file_logger('transformers.trainer', args.output_path, use_console=True)
    args.ip = utils.extract_ip()
    args.time= utils.get_time()
    logger.warning(f'>>> output in  {args.output_path}')
    logger.info(f'>>> using {args}')
    
    # 0. prepare tokenizer, model
    current_device = utils.get_local_rank2()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(current_device)
    preprocess = lambda example: tokenizer(
        example['input'] + tokenizer.sep_token + example['output'], 
        truncation=True, 
        padding=False, 
        max_length=512,
    )

    # 1. load data
    if args.data_path is None:
        args.data_path = schema.nli_data_path
    if args.dev_data_path is None:
        args.dev_data_path = schema.dev_nli_data_path
    logger.warning(f'>>> loading data: { args.data_path }')
    logger.warning(f'>>> loading dev data: { args.dev_data_path }')
    train_data, examples = modeling.load_data(args.data_path, mode='train-nli') 
    eval_data, eval_examples = modeling.load_data(args.dev_data_path, mode='dev-nli')
    examples = examples.map(preprocess)
    for example in examples:
        logger.info(f'>>> example:\n { tokenizer.decode(example["input_ids"]) }')
        logger.info(f'>>> tokenizer example: { example["input_ids"][:10] }...{ example["input_ids"][-10:]}')
        logger.info(f'>>> tokenizer labels: { example["labels"]}')
    train_data = train_data.shuffle().map(preprocess, num_proc=os.cpu_count())
    eval_data = eval_data.shuffle().map(preprocess, num_proc=os.cpu_count())

    # 2. prepare trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch,
            per_device_eval_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.total_batch//args.micro_batch,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epoch,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_strategy="epoch",
            logging_first_step=True, # convenient
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=2, # save best / last
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            output_dir=args.output_path,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.use_wandb else [],
            ignore_data_skip=args.ignore_data_skip,
        ),
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    if not args.use_wandb:
        wandb.init(mode='disabled')

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # save args
    utils.to_json(args.__dict__, f'{args.output_path}/train_args.json')

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(f'{args.output_path}/best')
    if args.use_wandb:
        wandb.config.update(args, allow_val_change=True)
        logger.info({'wandb-url': wandb.run.get_url()})

def train_text_classify(
    args,
):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    args.gradient_accumulation_steps = args.total_batch // args.micro_batch
    if ddp:
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size
    logger = utils.set_file_logger('transformers.trainer', args.output_path, use_console=True)
    args.ip = utils.extract_ip()
    args.time= utils.get_time()
    logger.warning(f'>>> output in  {args.output_path}')
    logger.info(f'>>> using {args}')
    
    # 0. prepare tokenizer, model
    current_device = utils.get_local_rank2()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(current_device)
    preprocess = lambda example: tokenizer(
        example['text'], 
        truncation=True, 
        padding=False, 
        max_length=128,
    )

    # 1. load data
    train_data, examples = modeling.load_attr_data(args.attr, mode='train-text') 
    eval_data, eval_examples = modeling.load_attr_data(args.attr, mode='dev-text')
    examples = examples.map(preprocess)
    for example in examples:
        logger.info(f'>>> example:\n { tokenizer.decode(example["input_ids"]) }')
        logger.info(f'>>> tokenizer example: { example["input_ids"][:10] }...{ example["input_ids"][-10:]}')
        logger.info(f'>>> tokenizer labels: { example["labels"]}')
    train_data = train_data.shuffle().map(preprocess, num_proc=os.cpu_count())
    eval_data = eval_data.shuffle().map(preprocess, num_proc=os.cpu_count())

    # 2. prepare trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch,
            per_device_eval_batch_size=args.micro_batch,
            gradient_accumulation_steps=args.total_batch//args.micro_batch,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epoch,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_strategy="epoch",
            logging_first_step=True, # convenient
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=2, # save best / last
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            output_dir=args.output_path,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.use_wandb else [],
            ignore_data_skip=args.ignore_data_skip,
        ),
        data_collator=transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    if not args.use_wandb:
        wandb.init(mode='disabled')

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # save args
    utils.to_json(args.__dict__, f'{args.output_path}/train_args.json')

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(f'{args.output_path}/best')
    if args.use_wandb:
        wandb.config.update(args, allow_val_change=True)
        logger.info({'wandb-url': wandb.run.get_url()})

def _start(
    *, 
    task_name: str='cvae-lyrical-only',
    data_path: str=None,
    dev_data_path: str=None,
    model_path:str='bert-base-uncased',
    cutoff: int=512,
    use_wandb:bool=False, 
    encoder_path:str='bert-base-uncased',
    decoder_path:str='microsoft/DialoGPT-medium',
    checkpoint_dir:str=None,
    attr:str=None,
    share_encoder:bool=False,
    full_decoder:bool=False,
    micro_batch:int=4,
    total_batch:int=32,
    warmup_ratio:float= 0.05,
    num_epoch:int=100,
    latent_size:int=64,
    learning_rate:float=5e-5,
    log_steps:int=4,
    int8:bool=False,
    ignore_data_skip:bool=False,
    resume_from_checkpoint:str=None,
    use_context:bool=False,
    type:str='',
    attrs:str=None, 
    ids:str=None,
    kl_ratio:float=0.,
    attr_cls_ratio:float=0.,
    attr_gap_ratio:float=0.,
    prior_cls: bool=False,
    prior_gap: bool=False,
    negap: bool=False
):
    import inspect
    frame = inspect.currentframe()
    names, _, _, locals = inspect.getargvalues(frame)
    args = utils.HyperParams().from_inspect(names, locals)
    # !important
    modeling.parse_attr(args)

    if args.type == 'latent':
        args.output_path = f'{args.checkpoint_dir}/{args.attr}-latent'
        train_latent_classify(args)
    elif args.type == 'text':
        args.output_path = f'outs/attribute-dialogue/{args.task_name}'
        train_text_classify(args)
    elif args.type == 'nli':
        args.output_path = f'outs/attribute-dialogue/{args.task_name}'
        train_nli_classify(args)
        # setattr(CustomTrainer,'compute_loss', xx)
        setattr(transformers.BertForSequenceClassification,'forward',focal_forward)
    else:
        raise Exception('cannot judge run latent or text')

if __name__ == "__main__":
    import defopt
    defopt.run(_start)
