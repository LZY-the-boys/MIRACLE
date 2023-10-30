import copy
import torch
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer,AutoModelForSequenceClassification, AutoModelForCausalLM
import math
import transformers
import utils
from datasets import Dataset
import tqdm
import sys
import gradio as gr
from transformers.trainer_callback import PrinterCallback
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from typing import Callable, List, Optional, Union
import evaluate
from collections import namedtuple
import numpy as np
import logging
import random
import spacy

class InferMiddleWare:
    
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 60,
            "num_beams": 4,
            # NOTE !
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "max_new_tokens": 1024,
            "min_new_tokens": 25,
            "do_sample": False,
            "repetition_penalty": 1,
            "max_memory": 1024,
            # "bad_words_ids": tokenizer(['<unk>','<s>'], add_special_tokens=False).input_ids,
            # "force_words_ids": tokenizer(['</s>'], add_special_tokens=False).input_ids, is_constraint_gen_mode can only use `is_contrastive_search_gen_mode`
        } 
        if kwargs:
            for n, v in kwargs.items():
                self.generation_config[n] = v
        # TODO:
        self.device = torch.device("cuda:0")
        self.is_encoder_decoder=False

    def generate_batched(
        self,
        input_ids: List[torch.Tensor],
        length_sampler= None,
        batch_size: int = 4,
        pad_to_multiple_of: int = None,
        **generation_kwargs,
    ):
        # input: tensor, output: tensor
        self.generation_config.update(generation_kwargs)
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(input_ids), batch_size)

        for i in tqdm.trange(0, len(input_ids), batch_size, desc='generating '):
            if length_sampler is not None:
                self.generation_config["max_new_tokens"] = length_sampler()
            end_index = min(len(input_ids), i + batch_size)

            batch = input_ids[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}
            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.device)

            if 'max_memory' in self.generation_config:
                self.generation_config.pop('max_memory')
            generations = self.model.generate(**padded_inputs, **self.generation_config)
            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt
                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

class TextClassifierMiddleWare:

    def __init__(self, classifier_dict, device='cuda:0', **kwargs):
        # pre_load classifer
        self.classifier_dict= classifier_dict
        self.device= device

    def compute_metrics(self, pred):
        labels = torch.tensor(pred.label_ids).long()
        preds = torch.softmax(torch.tensor(pred.predictions,dtype=float),dim=-1)
        # out[i][j] = preds[i][labels[i][j]]
        probs = torch.gather(preds, 1, labels.view(-1, 1))
        acc = torch.mean(probs).item()
        return {
            'scores': [prob.item() for prob in probs],
            'accuracy': round(acc,6)
        }
    
    def make_batch(self, sents, labels, name, sents2=None):
        
        tokenizer=self.classifier_dict[name]['tokenizer'] 
        dataset = Dataset.from_dict({
            'labels': labels,
            'text': sents,
        })
        if isinstance(name, str) and 'agnews-topic' in name:
            # TODO: change to sents2
            # 文本匹配
            topics = ["world","sports","business","science"]
            eval_dataset = dataset.map(lambda e: tokenizer(topics[e['labels']]+'[SEP]'+e['text'], truncation=True, padding='max_length', max_length=100))
            eval_dataset = eval_dataset.map(lambda e: {'labels': 1})
        elif name == 'nli':
            dataset = Dataset.from_dict({
                'input': sents,
                'output': sents2,
                'labels': labels,
            })
            eval_dataset = dataset.map(lambda example: tokenizer(
                example['input'] + tokenizer.sep_token + example['output'], 
                truncation=True, 
                padding=False, 
                max_length=512,
            ))
        else:
            eval_dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return eval_dataset

    def test_single(self, texts, labels, name, reduce_sum=True, texts2=None):
        eval_dataset = self.make_batch(texts, labels, name, texts2)

        if 'model' not in self.classifier_dict[name]:
            model = self.classifier_dict[name]['text']
        else:
            model = self.classifier_dict[name]['model']
        # avoid evaluate log
        trainer = transformers.Trainer(
            model=model,
            args = transformers.TrainingArguments(
                output_dir='outs',
                do_train = False,
                disable_tqdm=True,
                do_predict = True,
                per_device_eval_batch_size=8 if 'name' in ['nli'] else 128,
                dataloader_drop_last = False,
                report_to=[]
            ),
            compute_metrics=self.compute_metrics,
            data_collator=transformers.DataCollatorWithPadding(self.classifier_dict[name]['tokenizer'] ),
        )
        trainer.remove_callback(PrinterCallback)
        if reduce_sum:
            return trainer.evaluate(eval_dataset)['eval_accuracy']
        return {
            'scores': trainer.evaluate(eval_dataset)['eval_scores'],
            'acc': trainer.evaluate(eval_dataset)['eval_accuracy'],
        }

    def compute_acc(self, texts, labels, names, reduce_sum=True):
        result = []
        assert len(labels) == len(names)
        for label, name in zip(labels, names):
            result.append( self.test_single(
                texts,
                [label]*len(texts),
                name,
                reduce_sum,
            ))
        if reduce_sum:
            return {n:v for n,v in zip(names, result)}, None
        acc = {n:v['acc'] for n,v in zip(names, result)}
        acc_details = [ r['scores'] for r in result ]
        acc_details = [ list(d) for d in zip(*acc_details)]
        acc_details = [ {n:dd for dd, n in zip(d,names)} for d in acc_details]
        return acc , acc_details
            
    def compute_acc2(self, texts, labels, name, reduce_sum=True):
        assert len(texts) == len(labels)
        if isinstance(name, list):
            result = []
            for i, n in enumerate(name):
                result.append( self.test_single(
                    texts,
                    [label[i] for label in labels],
                    n,
                    reduce_sum,
                ))
            if reduce_sum:
                return {n:v for n,v in zip(name, result)}, None
            acc = {n:v['acc'] for n,v in zip(name, result)}
            acc_details = [ r['scores'] for r in result ]
            acc_details = [ list(d) for d in zip(*acc_details)]
            acc_details = [ {n:dd for dd, n in zip(d,name)} for d in acc_details]
            return acc , acc_details
        else:
            result =  self.test_single(
                texts,
                labels,
                name,
                reduce_sum,
            )
            if reduce_sum:
                return result, None
            return result['acc'], result['scores']

    def compute_NLI(self, history, response, reduce_sum=True):
        assert len(history) == len(response)
        result =  self.test_single(
            history,
            [1]*len(history),
            'nli', 
            reduce_sum,
            texts2=response,
        )
        if reduce_sum:
            return result, None
        return result['acc'], result['scores']

class GenerationMetricMiddleWare:

    def compute_bleu(self,label, pred):
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import numpy as np
        score = 0
        for weights in [
            (0.25, 0.25, 0.25, 0.25), # 默认 bleu4
            (0.33, 0.33, 0.33, 0), # 3
            (0.5, 0.5, 0, 0), # 2
            (1, 0, 0, 0), #1
        ]:
            score += np.mean(
                [sentence_bleu(
                    references=[list(a)], 
                    hypothesis=list(b),
                    smoothing_function=SmoothingFunction().method1,
                    weights=weights
                    ) for a, b in zip(label, pred)]
                )
        sentence_score = score/4
        corpus_score = corpus_bleu([[t] for t in pred], label)               
        return sentence_score, corpus_score

    def compute_lawrouge(self,label,pred):
        import lawrouge
        rouge = lawrouge.Rouge()
        # [。。]
        rouge.sentence_split = None
        # ['']
        for i,p in enumerate(pred):
            if p == '':
                pred[i] = '。'
        scores = rouge.get_scores(self, pred,label, avg=1)
        return scores

    def compute_rouge2(self,label, pred, weights=None, mode='weighted'):
        # Problem: cannot handle '.....' and ''
        import rouge
        rouge = rouge.Rouge()
        weights = weights or (0.2, 0.4, 0.4)
        if isinstance(label, str):
            label = [label]
        if isinstance(pred, str):
            pred = [pred]  
        # label = [' '.join(x) for x in label]
        # pred = [' '.join(x) for x in pred]
        pred = [ 'a' if l=='' or l=='.' else l for l in pred]
        rouge_dict = rouge.get_scores(hyps=pred, refs=label, avg=1)
        rouge_dict = {k: 100* v['f'] for k,v in rouge_dict.items()}
        return rouge_dict

    def compute_rouge(self, label, pred):
        from rouge_score import rouge_scorer
        rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
        rouge_score = {k:0 for k in rouge_metrics}
        scorer = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=True)
        for l ,p in zip(label, pred):
            rouge_score = { k: rouge_score[k]+ v.fmeasure for k,v in scorer.score(target=l, prediction=p).items() }
        rouge_score = {k: v/len(pred) for k,v in rouge_score.items()}
        return rouge_score

    def compute_bertscore(self,label,pred):
        import bert_score
        P, R, F1 = bert_score.score(pred, label, lang="zh", verbose=True)
        return F1.mean()

    def compute_distinct3(self, responses, num_sample=5):
        # len unique ngram / len words 
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        generations_batch = list(chunks(responses, num_sample))
        dist1, dist2, dist3 = [], [], []
        # calculate dist1, dist2, dist3 across generations for every prompt
        for generations in (generations_batch):
            unigrams, bigrams, trigrams = set(), set(), set()
            total_words = 0
            for gen in generations:
                o = gen.split(' ')
                total_words += len(o)
                unigrams.update(o)
                for i in range(len(o) - 1):
                    bigrams.add(o[i] + '_' + o[i + 1])
                for i in range(len(o) - 2):
                    trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
            dist1.append(len(unigrams) / total_words)
            dist2.append(len(bigrams) / total_words)
            dist3.append(len(trigrams) / total_words)

        # take the mean across prompts
        metrics = {"dist1":float(np.nanmean(dist1)),"dist2":float(np.nanmean(dist2)),"dist3":float(np.nanmean(dist3))}
        return metrics

    def compute_distinct(self, responses):
        # len unique ngram / len words 
        # compute in corpus level
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in responses:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        dist1=len(unigrams) / total_words
        dist2=len(bigrams) / total_words
        dist3=len(trigrams) / total_words
        # take the mean across prompts
        metrics = {"dist1":float(np.nanmean(dist1)),"dist2":float(np.nanmean(dist2)),"dist3":float(np.nanmean(dist3))}
        return metrics

    def compute_distinct2(self, responses):
        # len unique ngram / len words 
        # compute in sentence level 
        dist1, dist2, dist3 = [],[],[] 
        total_words = 0
        for gen in responses:
            o = gen.split(' ')
            total_words = len(o)
            unigrams, bigrams, trigrams = set(), set(), set()
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
            dist1.append(len(unigrams) / total_words)
            dist2.append(len(bigrams) / total_words)
            dist3.append(len(trigrams) / total_words)
        # take the mean across prompts
        metrics = {"dist1":float(np.nanmean(dist1)),"dist2":float(np.nanmean(dist2)),"dist3":float(np.nanmean(dist3))}
        return metrics

    def compute_ppl2(self, preds, prompts=None, model_name=f'{utils.MODEL_DIR}/gpt2-xl', reduce_sum=True):
        preds = [p for p in preds if p != '']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        perplexities = []
        if prompts is None:
            prompts = [None]*len(preds)
        for pred, prompt in tqdm.tqdm(zip(preds, prompts),total=len(preds), desc='PPL ', leave=False):
        
            full_input_ids = tokenizer.encode(pred, return_tensors='pt').to(device)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            if prompt is not None:
                # for every generation conditioned on the user_prompt
                prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
                loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
                ppl = math.exp(loss.item())
            else:
                loss = full_loss / full_input_ids.shape[1]
                ppl = math.exp(loss.item())

            if ppl < 1e4:   # for sanity
                perplexities.append(ppl)
        del model
        del tokenizer
        if reduce_sum:
            return np.nanmean(perplexities)
        else:
            return perplexities

    def compute_ppl(self, pred):
        if '' in pred: # len must >= 1
            return 0
        perplexity = evaluate.load("perplexity", module_type="metric")
        results = perplexity.compute(predictions=pred, model_id=f'{utils.MODEL_DIR}/gpt2-xl')
        return results['mean_perplexity']

    def compute_sbleu(self,texts):
        senbleu = 0 
        texts = [text.split(' ') for text in texts]
        for i in tqdm.trange(len(texts), desc='calc senblue',leave=False):
            # ref, hyp
            senbleu += sentence_bleu(
                references=random.sample(texts[:i]+texts[i+1:],150), 
                hypothesis=texts[i], 
                smoothing_function=SmoothingFunction().method1
            )
        senbleu = senbleu/len(texts)
        return senbleu