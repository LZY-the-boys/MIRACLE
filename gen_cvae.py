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
import pandas as pd
import numpy as np
from tqdm import tqdm


def text_metrics( args, history, response, gold, ids=None,postfix='' ):

    gen_metric_mid = middleware.GenerationMetricMiddleWare()
    score_metric_mid = middleware.TextClassifierMiddleWare(args.scorer_list)
    self_blue = gen_metric_mid.compute_sbleu(response)
    sentence_blue, corpus_blue = gen_metric_mid.compute_bleu(gold,response)
    rouge_score = gen_metric_mid.compute_rouge(gold,response)
    ppl_score2 = gen_metric_mid.compute_ppl2(response) 
    corpus_distinct_score = gen_metric_mid.compute_distinct(response)
    sentence_distinct_score =gen_metric_mid.compute_distinct2(response)
    
    nli, nli_details = score_metric_mid.compute_NLI(history, response, reduce_sum=False)
    if ids is not None:
        accs, acc_details = score_metric_mid.compute_acc2(response, ids, args.attrs, reduce_sum=False)
    metrics = {
        'sentence_blue': sentence_blue*100,
        # 'corpus_blue': corpus_blue,
        'ppl': ppl_score2,
        'rouge_score': utils.mean(rouge_score)*100,
        'distinct_score': utils.mean([utils.mean(corpus_distinct_score)*100, utils.mean(sentence_distinct_score)*100]),
        'self_blue': self_blue*100,
        'nli': nli*100,
        'acc_sum': utils.mean(accs)*100,
    }
    metrics.update({f'acc{k}': accs[k]*100 for k in args.attrs})
    utils.to_json({
        'details':[  
            {'history': r[0], 'label': r[1],'gen': r[2], 'score': r[3], 'nli': r[4], 'tag':r[5]} 
            for r in zip(history,gold, response, acc_details, nli_details, ids)],
        'metrics':  metrics,
        }, f'{args.attr_output_path}/{args.task_name}-gen{postfix}.json'
    )
    return metrics

def gen_attr(model, tokenizer, decoder_tokenizer, dataloader, args, logger, name=None, forward=False):
    prefix = 'dev'
    if name is None: 
        name='' 
        prefix='test'
    if args.debug: name = str(name) + '-debug'
    else: name = str(name)

    if not hasattr(args, 'scorer_list'):
        modeling.load_scorer(args)
    gen_mid = modeling.VAEInferMiddleWare(model, tokenizer, decoder_tokenizer)
    gen_metric_mid = middleware.GenerationMetricMiddleWare()
    score_metric_mid = middleware.TextClassifierMiddleWare(args.scorer_list)

    sample_kwargs = {}
    dev_metrics = {}
    # NOTE only one epoch
    for id in args.ids:
        sample_kwargs = {
            'algorithm': args.algorithm,
            'ids': id,
            'scorers': args.scorer_list,
        }
        # must be same all across the training process
        attr_name = '_'+','.join([str(i) for i in id])+'_a'+str(os.environ.get('atol',1e-3))+"r"+str(os.environ.get('rtol',1e-3))
        for _ , batch in enumerate((dataloader)):
            
            # NOTE input_ids need to be list[tensor], cannot pad!
            input_tensors, input_history, response_labels = batch["input_ids"], batch['history'], batch['label_texts']

            # NOTE test gen
            if args.use_context:
                response_tensors = gen_mid.generate_batched(
                    input_tensors,
                    decoder_input_ids=batch['decoder_input_ids'],
                    step_batch_size=4,
                    **sample_kwargs,
                )
            else:
                response_tensors = gen_mid.generate_batched(
                    input_tensors,
                    step_batch_size=4,
                    **sample_kwargs,
                )
            response_texts = decoder_tokenizer.batch_decode(response_tensors, skip_special_tokens=not args.debug)

            # generation metrics
            sentence_blue, corpus_blue = gen_metric_mid.compute_bleu(response_labels,response_texts)
            rouge_score = gen_metric_mid.compute_rouge(response_labels,response_texts)
            utils.gpu_clean()
            ppl_score2 = gen_metric_mid.compute_ppl2(response_texts) 
            distinct_score = gen_metric_mid.compute_distinct(response_texts)
            self_blue = gen_metric_mid.compute_sbleu(response_texts)
            accs, acc_details = score_metric_mid.compute_acc(response_texts, id, args.attrs, reduce_sum=False)
            nli, nli_score = score_metric_mid.compute_NLI(input_history, response_texts,reduce_sum=False)
            dev_metrics[attr_name] = {
                'sentence_blue': sentence_blue,
                'corpus_blue': corpus_blue,
                'ppl': ppl_score2,
                'rouge_score': rouge_score,
                'distinct_score': distinct_score,
                'self_blue': self_blue,
                'nli': nli,
                'accs': accs,
                'avg_acc': utils.mean(accs),
                'wandb': wandb.run.get_url() if args.use_wandb else None,
            }
        logger.warning(dev_metrics)
        if prefix == 'test':
            utils.to_jsonl([{ args.task_name+attr_name : dev_metrics }], f'{args.output_path}/test_metrics.json', mode='a')
        utils.to_json({
            'details':[{'history': r[0], 'label': r[1],'gen': r[2], 'score': r[3], 'nli': r[4]} for r in zip(input_history,response_labels, response_texts, acc_details, nli_score)],
            'metrics': dev_metrics,
            },f'{args.attr_output_path}/gen{attr_name}.json'
        )
        logger.info(f'>>> generating to {args.attr_output_path}/gen{attr_name}.json')
    
        # log to wandb
        # NOTE dev or test
        wandb_metrics = {f'{prefix}/{n}':v for n,v in dev_metrics.items()}
        if prefix == 'test':
            step = os.environ.get('WANDB_STEP', 0)
            if step == '':
                step = 0
            wandb.log(wandb_metrics, step=int(step))
        else:
            wandb.log(wandb_metrics)

def gen_attr2(model, tokenizer, decoder_tokenizer, dataloader, args, logger, name=None, forward=False):
    prefix = 'dev'
    if name is None: 
        name=args.postfix 
    if args.debug: name = str(name) + '-debug'
    else: name = str(name)

    if not hasattr(args, 'scorer_list'):
        # modeling.load_scorer(args)
        modeling.load_scorer(args, use_extern_latent=False)
        for attr in args.attrs:
            args.scorer_list[attr]['latent'] = model.latent_cls[attr]
    
    gen_mid = modeling.VAEInferMiddleWare(model, tokenizer, decoder_tokenizer)

    sample_kwargs = {}
    dev_metrics = {}
    # NOTE only one epoch
    sample_kwargs = {
        'algorithm': args.algorithm,
        'scorers': {k:v for k,v in args.scorer_list.items() if k!= 'nli'},
    }
    # must be same all across the training process
    for _ , batch in enumerate((dataloader)):
        
        # NOTE input_ids need to be list[tensor], cannot pad!
        input_tensors, input_history, response_labels, tags = batch["input_ids"], batch['history'], batch['label_texts'], batch['tag']

        # NOTE test gen
        if args.use_context:
            response_tensors = gen_mid.generate_batched(
                input_tensors,
                decoder_input_ids=batch['decoder_input_ids'],
                step_batch_size=4,
                ids = tags,
                **sample_kwargs,
            )
        else:
            response_tensors = gen_mid.generate_batched(
                input_tensors,
                step_batch_size=4,
                ids = tags,
                **sample_kwargs,
            )
        response_texts = decoder_tokenizer.batch_decode(response_tensors, skip_special_tokens=not args.debug)

        # generation metrics
        metrics = text_metrics(args, input_history, response_texts, response_labels, tags, postfix=os.environ.get('z_prior_type','standard'))
        logger.warning(metrics)
        columns = ['acc_sum','acc0','acc1','acc2','sentence_blue', 'rouge_score', 'nli','ppl','distinct_score','self_blue']
        path = f'{args.attr_output_path}/result.xlsx'
        df = pd.DataFrame(metrics, index=[name]).reindex(columns=columns)
        if os.path.exists(path):
            previous = pd.read_excel(path,index_col=0)
            df = pd.concat([previous,df])
        df.to_excel(path,index=True)

def gen(model, tokenizer, decoder_tokenizer, dataloader, args, logger, name=None,forward=False):

    prefix = 'dev'
    if name is None: 
        name=''
        prefix='test'
    if args.debug: name = str(name) + '-debug'

    gen_mid = modeling.VAEInferMiddleWare(model, tokenizer, decoder_tokenizer)
    gen_metric_mid = middleware.GenerationMetricMiddleWare()

    # NOTE only one epoch
    for _ , batch in enumerate((dataloader)):
        
        # NOTE input_ids need to be list[tensor], cannot pad!
        input_tensors, input_history, response_labels = batch["input_ids"], batch['history'], batch['label_texts']
        # NOTE test forward
        dev_metrics = {}
        if forward:
            if args.use_context:
                output, ppl = gen_mid.forward_batched(
                    input_tensors,
                    full_ids=batch['full_ids'],
                    label_ids=batch['label_ids'],
                    step_batch_size=4,
                )
                raw_forward_texts = decoder_tokenizer.batch_decode(output, skip_special_tokens=False)
                forward_texts = [r.split('\n\nAssistant:')[-1] for r in raw_forward_texts]
            else:
                output, ppl = gen_mid.forward_batched(
                    input_tensors,
                    full_ids=batch['label_ids'],
                    label_ids=batch['label_ids'],
                    step_batch_size=4,
                )
                forward_texts = decoder_tokenizer.batch_decode(output, skip_special_tokens=False)
            
            sentence_blue, corpus_blue = gen_metric_mid.compute_bleu(response_labels,forward_texts)
            rouge_score = gen_metric_mid.compute_rouge(response_labels,forward_texts)
            distinct_score = gen_metric_mid.compute_distinct(forward_texts)
            self_blue = gen_metric_mid.compute_sbleu(forward_texts)
            dev_metrics = {
                'forward':{
                    'sentence_blue': sentence_blue,
                    'corpus_blue': corpus_blue,
                    'ppl': ppl,
                    'rouge_score': rouge_score,
                    'distinct_score': distinct_score,
                    'self_blue': self_blue,
                }
            }
            logger.info(dev_metrics['forward'])
            utils.to_json([{'label': r[0],'gen': r[1]} for r in zip(response_labels, forward_texts)], f'{args.output_path}/forward{name}.json')

        # NOTE test gen
        if args.use_context:
            response_tensors = gen_mid.generate_batched(
                input_tensors,
                decoder_input_ids=batch['decoder_input_ids'],
                step_batch_size=4,
            )
        else:
            response_tensors = gen_mid.generate_batched(
                input_tensors,
                step_batch_size=4,
            )
        response_texts = decoder_tokenizer.batch_decode(response_tensors, skip_special_tokens=not args.debug)

        # generation metrics
        sentence_blue, corpus_blue = gen_metric_mid.compute_bleu(response_labels,response_texts)
        rouge_score = gen_metric_mid.compute_rouge(response_labels,response_texts)
        ppl_score2 = gen_metric_mid.compute_ppl2(response_texts) 
        distinct_score = gen_metric_mid.compute_distinct(response_texts)
        self_blue = gen_metric_mid.compute_sbleu(response_texts)
        # entailment = gen_metric_mid.compute_NLI(input_history, response_texts) # TODO:
        dev_metrics.update({
            'gen':{
                'sentence_blue': sentence_blue,
                'corpus_blue': corpus_blue,
                'ppl': ppl_score2,
                'rouge_score': rouge_score,
                'distinct_score': distinct_score,
                'self_blue': self_blue,
            }
        })
        logger.info(dev_metrics['gen'])
        if prefix == 'test':
            utils.to_json({args.task_name:dev_metrics},f'{args.output_path}/test_metrics.json')
            utils.to_json([{'label': r[0],'gen': r[1]} for r in zip(response_labels, response_texts)], f'{args.output_path}/gen{name}.json')
        else: # too much
            os.makedirs(f'{args.output_path}/tmp', exist_ok=True)
            utils.to_json([{'label': r[0],'gen': r[1]} for r in zip(response_labels, response_texts)], f'{args.output_path}/tmp/gen{name}.json')
        
        # log to wandb
        # NOTE dev or test
        wandb_metrics = {f'{prefix}/{n}':v for n,v in dev_metrics.items()}
        # wandb_metrics['forward-text'] = wandb.Table(
        #     columns=["label", "gen"], 
        #     rows=[list(r) for r in zip(response_labels, forward_texts)] 
        # )
        wandb_metrics['gen-text'] = wandb.Table(
            columns=["label", "gen"], 
            rows=[list(r) for r in zip(response_labels, response_texts)] 
        )
        # currently not support continuous logging
        if prefix == 'test':
            step = os.environ.get('WANDB_STEP', 0)
            if step == '':
                step = 0
            wandb.log(wandb_metrics, step=int(step))
        else:
            wandb.log(wandb_metrics)

def cvae_start(
    *, 
    task_name: str='cvae-lyrical-only',
    checkpoint_dir: str='outs/attribute-dialogue/cvae-lyrical-only/',
    cutoff: int=512,
    use_context: bool=False,
    data_path: str=None,
    dev_data_path: str=None, # use this
    encoder_path:str='bert-base-uncased',
    decoder_path:str='microsoft/DialoGPT-medium',
    share_encoder:bool=False,
    full_decoder:bool=False,
    algorithm: str=None,
    attrs: str=None, # 0,1,2
    ids: str=None, # 0,0;0,1;1,0;1,1
    use_wandb:bool=False, 
    postfix: str=None,
    micro_batch:int=4,
    share:bool=False,
    latent_size:int=64,
    int8:bool=False,
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
    args.output_path = args.checkpoint_dir
    args.debug = utils.DEBUG
    modeling.parse_attr(args)

    # args_path = (os.path.join(args.checkpoint_dir, 'train_args.json'))
    # if os.path.exists(args_path):
    #     training_args = utils.from_json(args_path)
    #     args.share = training_args['share_encoder']
    #     args.latent_size = training_args['latent_size']
    
    logger = utils.set_file_logger(__name__, args.output_path, use_console=True)
    args.ip = utils.extract_ip()
    args.time= utils.get_time()
    logger.warning(f'>>> using {args}')

    # 0. prepare tokenizer
    tokenizer, decoder_tokenizer = modeling.load_tokenizer(args)
    # 1. load data
    prompter = prompt.attribute_dialogue(tokenizer, decoder_tokenizer, args.cutoff, decoder_add_eos=True, decoder_add_bos=True)
    if args.use_context:
        preprocess = prompter.preprocess_gen_context
    else:
        preprocess = prompter.preprocess_gen
    data, examples = modeling.load_data(schema.dev_cvae_data_path, mode='gen', num=4 if args.debug else 0 )
    # 1.1 test example
    examples = examples.map(preprocess)
    for example in examples:
        logger.info(f'>>> prior_input_ids example:\n { tokenizer.decode(example["input_ids"]) }')
        logger.info(f'>>> tokenizer example: { example["input_ids"][:10] }...{ example["input_ids"][-10:]}')
        logger.info(f'>>> tokenizer label: { decoder_tokenizer.decode([0 if l==-100 else l for l in example["label_ids"]])}')
        logger.info(f'>>> tokenizer label_ids: { example["label_ids"][:10]}...{example["label_ids"][-10:]}')
    # 1.2 process datas
    data = data.map(preprocess, num_proc=os.cpu_count())
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=len(data),
        collate_fn= lambda data: dict((key, [ torch.tensor(d[key]) if 'ids' in key else d[key] for d in data]) for key in data[0]),
        shuffle=False,
    )
    # 2. prepare model
    model = modeling.load_model(args, decoder_tokenizer)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.postfix if args.postfix else 'checkpoint-final')
    logger.warning(f'>>> load trained checkpoint from : { checkpoint_path }')
    state_dict = torch.load(checkpoint_path)
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k.replace('_orig_mod.','')] = state_dict[k]
    del state_dict
    model.load_state_dict(new_state_dict,strict=False)

    if args.use_wandb:
        wandb.init(resume="allow")
    else:
        wandb.init(mode='disabled')

    if args.algorithm is None:
        args.attr_output_path = f'{args.output_path}/no-personlized'
        os.makedirs(args.attr_output_path, exist_ok=True)
        gen_attr2(model, tokenizer, decoder_tokenizer, dataloader, args, logger)
    else:
        args.attr_output_path = f'{args.output_path}'
        os.makedirs(args.attr_output_path, exist_ok=True)
        gen_attr2(model, tokenizer, decoder_tokenizer, dataloader, args, logger)
        
if __name__ == "__main__":
    import defopt
    defopt.run(cvae_start)