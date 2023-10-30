import math
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
from argparse import Namespace
import copy
from transformers import (
    BertModel,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoModelWithLMHead,
    get_polynomial_decay_schedule_with_warmup, 
    GenerationConfig,
)
import transformers
import pandas as pd
import random
import utils
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model, CausalLMOutputWithCrossAttentions, GPT2PreTrainedModel,GPT2LMHeadModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration,T5Stack
from transformers import LlamaForCausalLM,LlamaTokenizer,BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
import tqdm
import wandb
import middleware
from datasets import load_dataset, Dataset, disable_caching
import datasets;datasets.config.IN_MEMORY_MAX_SIZE=262144
disable_caching()
import logging
import schema

class CVAE(nn.Module):
    kl_threshold = 0.9
    beta = 0.

    def __init__(self, prior_encoder, posterior_encoder, decoder, args):
        super().__init__()
        self.prior_encoder = prior_encoder
        self.posterior_encoder = posterior_encoder
        self.decoder = decoder
        self.args = args

    def reparameterize(self, mu, logvar, nsamples=1):
        mu_ = mu.unsqueeze(1).repeat(1, nsamples, 1)
        std_ = torch.exp(0.5*logvar).unsqueeze(1).repeat(1, nsamples, 1)
        eps = torch.zeros_like(std_).normal_()
        # [ batch, nsamples, latent]
        return mu_ + eps * std_

    def Guassian_KL_distance(self, prior_logvar, posterior_logvar, prior_mean, posterior_mean):
        # NOTE logvar = log(std^2)
        # two normal distribute kl distance
        kl = 0.5 * (
            prior_logvar - posterior_logvar - 1
        +  (posterior_logvar - prior_logvar).exp()
        +  (prior_mean - posterior_mean).pow(2) / prior_logvar.exp()
        )
        # [batch, latent]
        kl_nomask = torch.sum(kl, dim=1).mean()
        kl_mask = None
        if self.kl_threshold:
            _mask = ( kl > self.kl_threshold ).float()
            kl_mask = ( _mask * kl ).sum(dim=1).mean() 
        return kl_nomask, kl_mask

    def forward(
        self, 
        prior_input_ids, 
        posterior_input_ids, 
        prior_attention_mask,
        posterior_attention_mask,
        decoder_input_ids, 
        labels, 
        **kwargs,
    ):

        prior_mean, prior_logvar = self.prior_encoder(
            input_ids=prior_input_ids, 
            attention_mask=prior_attention_mask
        )
        posterior_mean, posterior_logvar = self.posterior_encoder(
            input_ids=posterior_input_ids, 
            attention_mask=posterior_attention_mask
        )

        if self.training:
            if self.beta == 0:
                z_sample = posterior_mean
            else:
                z_sample = self.reparameterize(posterior_mean, posterior_logvar, nsamples=1).squeeze(1)
        else:
            # eval
            z_sample = prior_mean
        
        loss_kl_nomask, loss_kl = self.Guassian_KL_distance(
            prior_logvar, posterior_logvar, 
            prior_mean, posterior_mean
        )
        outputs = self.decoder(
            input_ids= decoder_input_ids, 
            z=z_sample, 
            labels=labels, 
        )
        logits = outputs.logits 
        loss_rec = outputs.loss
        loss = loss_rec + self.args.kl_ratio * self.beta * loss_kl

        extra = {
            'loss_rec': loss_rec.item(),
            'loss_kl': loss_kl.item(),
            'loss_kl_nomask': loss_kl_nomask.item(),
            'beta': self.beta,
            'mean': z_sample.mean().item(),
            'std': z_sample.std().item(),
        }
        # global variable
        self.args.extra = extra

        return {
            "loss":loss,
        }

    def sample_z(
        self, 
        input_ids, 
        attention_mask, 
        mode='prior',
    ):

        if 'raw' in mode:
            return torch.randn(*[input_ids.size(0) , self.args.latent_size]).to(self.decoder.device)
        
        if 'prior' in mode:
            mean, logvar = self.prior_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask,
            )
        elif 'posterior' in mode:
            mean, logvar = self.posterior_encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask,
            )
        else:
            raise Exception('no implementation')

        if 'msample' in mode:
            z = self.reparameterize(mean, logvar, nsamples=3) # [batch,3,latent]
            z = z.view(-1, z.size(-1))
        elif 'sample' in mode:
            z = self.reparameterize(mean, logvar, nsamples=1).squeeze(1)
        else:
            z = mean

        return z, mean, logvar

class AttrCVAE(CVAE):
    _keys_to_ignore_on_save=None # for resuming training
    kl_threshold = 0.9
    beta = 0.
    cls_threshold = 0.1
    mid_size = 128

    def __init__(self, prior_encoder, posterior_encoder, decoder, args):
        super().__init__(prior_encoder, posterior_encoder, decoder, args)
        assert isinstance(args.attrs, list) 
        self.latent_cls = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.latent_size, self.mid_size),
                nn.ReLU(),
                # TODO: to be fixed
                nn.Linear(self.mid_size, 2)
            ) for _ in range(3)]
        )
        # ) for attr in args.attrs]
        self.latent_cls.requires_grad_(False)
        for attr in args.attrs:
            self.latent_cls.requires_grad_(True)
        self.mean_cache = { k: None for k in args.attrs } 
        if self.args.prior_gap:
            self.prior_mean_cache = { k: None for k in args.attrs } 

    def latent_gap(
        self, 
        z,
        attr,     
        mean_cache,   
    ):
        batch_center = torch.mean(z, dim=0)
        loss = torch.tensor(0,device=batch_center.device,dtype=batch_center.dtype)

        for a in (self.args.attrs):
            if a != attr and mean_cache[a] is not None:
                loss += F.mse_loss(batch_center, mean_cache[a])
        
        mean_cache[attr] = batch_center.detach()
        if self.args.negap:
            return -loss
        return loss

    def latent_classify(
        self,
        z,
        tags,
        attr,
    ):
        logits = self.latent_cls[attr](z)
        loss = torch.nn.functional.cross_entropy(logits, tags)
        return loss

    def forward(
        self, 
        prior_input_ids, 
        posterior_input_ids, 
        prior_attention_mask,
        posterior_attention_mask,
        decoder_input_ids, 
        labels, 
        tags,
        attr:int,
        **kwargs,
    ):
        prior_mean, prior_logvar = self.prior_encoder(
            input_ids=prior_input_ids, 
            attention_mask=prior_attention_mask
        )
        posterior_mean, posterior_logvar = self.posterior_encoder(
            input_ids=posterior_input_ids, 
            attention_mask=posterior_attention_mask
        )

        if self.training:
            if self.beta == 0:
                z_sample = posterior_mean
            else:
                z_sample = self.reparameterize(posterior_mean, posterior_logvar, nsamples=1).squeeze(1)
            priorz_sample = self.reparameterize(prior_mean, prior_logvar, nsamples=1).squeeze(1)
        else:
            # eval
            z_sample = prior_mean
        
        loss_kl_nomask, loss_kl = self.Guassian_KL_distance(
            prior_logvar, posterior_logvar, 
            prior_mean, posterior_mean
        )
        outputs = self.decoder(
            input_ids= decoder_input_ids, 
            z=z_sample, 
            labels=labels, 
        )
        logits = outputs.logits 
        loss_rec = outputs.loss
        loss = loss_rec + self.args.kl_ratio * self.beta * loss_kl

        extra = {
            'loss_rec': loss_rec.item(),
            'loss_kl': loss_kl.item(),
            'loss_kl_nomask': loss_kl_nomask.item(),
            'beta': self.beta,
            'mean': z_sample.mean().item(),
            'std': z_sample.std().item(),
        }
        # global variable
        self.args.extra = extra

        loss_attr_cls = self.latent_classify(
            z_sample, 
            tags, 
            attr
        )
        if self.args.prior_cls:
            loss_attr_cls += self.latent_classify(
                priorz_sample, 
                tags, 
                attr
            )
        # cls_threshold
        if loss_attr_cls.item() < self.cls_threshold:
            loss += loss_attr_cls * 0.05
        else:
            loss += loss_attr_cls * self.args.attr_cls_ratio
        extra["loss_attr_cls"] = loss_attr_cls.item()

        loss_attr_gap = self.latent_gap(
            z_sample, 
            attr,
            self.mean_cache
        )
        if self.args.prior_gap:
            loss_attr_gap += self.latent_gap(
                priorz_sample, 
                attr,
                self.prior_mean_cache,
            )
        loss += loss_attr_gap * self.args.attr_gap_ratio
        extra["loss_attr_gap"] = loss_attr_gap.item()

        return {
            "loss":loss,
        }

class LinearClassifier(nn.Module):
    def __init__(self,input_dim,out_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(input_dim, input_dim),
            # nn.ReLU(),
            # nn.Linear(input_dim,out_dim),
            nn.Linear(input_dim, 4*input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4*input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim,out_dim),
        )
        self.num_classes = out_dim

    def forward(self,input_ids,labels=None):
        logits = self.net(input_ids)
        loss = None
        if labels is not None:
            if self.num_classes > 1:
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
            else:
                loss = torch.mean(torch.norm(logits - labels[:, None], dim=1) ** 2 * 0.5)
        # trainer: either dict or __get_item__
        return {
            'loss': loss,
            'logits': logits,
        }

class DiffEqualSampler(nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, scorers, device='cuda:0',weights=None, z_prior_type='standard'):
        super().__init__()
        if not scorers or len(scorers) == 0:
            raise Exception('wrong scorers')
        self.scorers = scorers
        self.current_device = device
        # standard N(0,I) or normal N(mu,varI)
        self.z_prior_type = z_prior_type
        if weights is None:
            self.weights = [1]*len(self.scorers)
        else:
            self.weights = weights
        self.logger = logging.getLogger('__main__')

    # ---------ode-------------------
    def forward(self, t, z):
        beta = self.beta_0 + t * (self.beta_1 - self.beta_0)
        z.requires_grad_(True)
        with torch.set_grad_enabled(True):
            cond_energy_negative = self.conditional_ebm_score(z)
            # dz/dt = \delta Energy; z_{t} - z_{t+1} = - dz/dt = - \delta Energy; (ODEsample t=T->0)
            score = torch.autograd.grad(cond_energy_negative.sum(), [z])[0]
            if self.z_prior_type == 'standard':
                dz_dt = -0.5 * beta * score
            # assume that var and mu is time invariant
            # if self.z_prior_type == 'normal' and os.environ.get('z_sample_mode','prior') == 'prior':
            #     dz_dt = -0.5 * beta * ( z + score )
            elif self.z_prior_type == 'normal':
                dz_dt = -0.5 * beta * ( ((self.var - 1) *z + self.mu) / self.var + score )
            else:
                raise Exception('wrong')
        self.steps += 1
        self.energy = cond_energy_negative
        wandb.log({
            'softmax': sum(self._energy_detail).exp().mean().item(), # = softmax(f(z), c)
            'logsoftmax': sum(self._energy_detail).mean().item(),
            'dz_dt': dz_dt.abs().mean().item(),
        })
        return dz_dt

    # get -Energy
    # energy = sum( - softmax(f(z),c) or (f(z) - c)^2 ) the smaller the better(-> 0 and > 0)
    def conditional_ebm_score(
        self, 
        z, 
    ): 
        energys = []
        for i, scorer, weight in zip(range(len(self.scorers.values())), self.scorers.values(), self.weights):
            label = torch.tensor(self.label)[:, i].to(z.device)
            func = scorer['latent']
            try:
                logits = func(z)['logits'] # custom classifier
            except:
                logits = func(z) # cvae classifier
            n_classes = logits.size(-1)
            if n_classes > 1:
                e = torch.gather(logits, 1, label.view(-1, 1)) - logits.logsumexp(1)
                # e = logits[:,id] - logits.logsumexp(1)
                # e = F.log_softmax(logits,dim=-1)[:, id]
                energys.append( weight * e )
            else:
                assert n_classes == 1, n_classes
                sigma = 0.1  # From previous work
                batch_id = torch.tensor(id).repeat(*logits.shape)
                e = - torch.norm(logits - batch_id , dim=1) ** 2 * 0.5 / (sigma ** 2)
                energys.append( weight * e )
        
        # EBM： Energy(C|z) = \sum_i Energy(c_i|z) <=> \PI softmax(f(z_i),c) (POE)
        self._energy_detail=energys
        total_energy = sum(energys)
        return total_energy 

    def ode_sample(self, z, label, t1=1.,t0=0.,beta_min=0.1,beta_max=20.,dt=1e-2, mu=None, var=None):
        from torchdiffeq import odeint_adjoint, odeint
        assert self.scorers, 'you have to set attr scorers first!'
        if self.z_prior_type == 'normal':
            assert mu is not None
            assert var is not None
            self.mu = mu.to(self.current_device)
            self.var = var.to(self.current_device)

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.label = label
        # -1 => 0
        z = z.to(self.current_device)
        batch_t = torch.linspace(t1, t0, 2).type(torch.float32).to(self.current_device) 
        self.steps = 0
        # NOTE calc z_0
        # relative `rtol` and absolute `atol` error tolerance.
        # if the error estimate is greater than some tolerance, then the step is redone with a smaller step size, and this repeats until the error is smaller than the provided tolerance
        list_z = odeint(
        # list_z = odeint_adjoint(
            self,  # self.forward dz/dt
            z,  # z_T
            batch_t, 
            atol=float(os.environ.get('atol',1e-3)), 
            rtol=float(os.environ.get('rtol',1e-3)),
            method='dopri5', 
        )
        # dopri8 cannot work; will stuck
        # dopri5 + atol 1e-9 hard to sample
        delta_z = list_z[-1] - z
        if os.environ.get('EBM_LOG', False):
            self.logger.info({
                'step': self.steps, 
                'delta_z': delta_z.abs().mean().item(), 
                'logsoftmax': self.energy.mean().item(), 
                'softmax': self.energy.exp().mean().item(),
                'detail': [e.exp().mean().item() for e in self._energy_detail]
            })
        return list_z

    # ---------sde------------------
    # dz/dt = - 0.5 beta(t) (x + 2*score) dt + sqrt(beta) dw
    def f(self, _t, z):
        t = -_t
        beta = self.beta_0 + t * (self.beta_1 - self.beta_0)
        z.requires_grad_(True)
        with torch.set_grad_enabled(True):
            energy_neg = self.sde_conditional_score(z)
            score = torch.autograd.grad(energy_neg.sum(), [z])[0]
            dz_dt = - 0.5 * beta * z - beta * score
        self.steps += 1
        return -dz_dt

    def g(self, _t, z):
        t = -_t
        beta = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return -torch.sqrt(beta)[None,None].repeat(*z.shape)

    def sde_sample(self, z, dt=1e-2, tweedie_correction=True):
        import torchsde
        self.CCF.eval()

        t = torch.tensor([-self.t1, -self.t0], device=self.device)
        zs = torchsde.sdeint(self, z.flatten(start_dim=1), t, dt=dt) # self.f , self.g
        zs = zs.view(len(t), *z.size())
        if tweedie_correction:
            zs[-1] = self.tweedie_correction(self.t0, zs[-1], dt)
        return zs   
    
    def tweedie_correction(self, t, z, dt):
        z.requires_grad_(True)
        with torch.set_grad_enabled(True):
            energy_neg = self.CCF.sde_conditional_score(z)
            score = torch.autograd.grad(energy_neg.sum(), [z])[0]
        return z + dt ** 2 * score

    def ld_sample(self, z, eps=1e-2, eps_decay=0.9, temperature=1.0, sample_steps=200, N=1024):
        z_t, z_sequence = z, [z]
        # sgld
        for t in range(sample_steps):
            cond_energy_neg = self.conditional_ebm_score(z_t)
            score = torch.autograd.grad(cond_energy_neg.sum(), [z_t])[0]
            z_t = z_t + eps / 2 * score + torch.sqrt(torch.tensor(eps)) * temperature * torch.randn_like(z_t)
            z_sequence.append(z_t)
            eps *= eps_decay
        return z_sequence

class VAEInferMiddleWare(middleware.InferMiddleWare):

    def __init__(self, model, tokenizer, decoder_tokenizer, **kwargs):
        # tokenizer is decoder_tokenizer
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 60,
            "num_beams": 1,
            "bos_token_id": decoder_tokenizer.bos_token_id,
            "eos_token_id": decoder_tokenizer.eos_token_id,
            "pad_token_id": decoder_tokenizer.pad_token_id,
            "max_new_tokens": 128,
            "min_new_tokens": 2,
            "do_sample": False,
            # "repetition_penalty": 1,
            "max_memory": 512,
            # "bad_words_ids": decoder_tokenizer(['<unk>','<s>'], add_special_tokens=False).input_ids,
            # "force_words_ids": tokenizer(['</s>'], add_special_tokens=False).input_ids, is_constraint_gen_mode can only use `is_contrastive_search_gen_mode`
        } 
        self.tokenizer = decoder_tokenizer
        self.encoder_tokenizer = tokenizer
        self.model = model
        self.is_encoder_decoder = True
        self.current_device = utils.get_local_rank2()
        self.algorithm=None

    # can be used in latent classifier
    @torch.no_grad()
    def sample_batched(
        self,
        input_ids: list[torch.Tensor],
        batch_size,
        only_z = True,
    ):
        zs,zmeans,zlogvars = [],[],[]
        # in case we have fewer examples than bs
        batch_size = min(len(input_ids), batch_size)

        for i in tqdm.trange(0, len(input_ids), batch_size, desc='sampling ', leave=False):

            end_index = min(len(input_ids), i + batch_size)
            batch = input_ids[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}
            padded_inputs = self.encoder_tokenizer.pad(
                inputs,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.current_device)
            z, mean, logvar = self.model.sample_z(**padded_inputs, mode=os.environ.get('z_sample_mode','prior'))
            # must be same size
            # for generation, mask in zip(output, padded_inputs["attention_mask"]):
            #         output = generation[(1 - mask).sum() :]  # remove padding
            zs.append(z.cpu()) # save vram
            zmeans.append(mean.cpu()) # save vram
            zlogvars.append(logvar.cpu()) # save vram
        zs = torch.cat(zs)
        zmeans = torch.cat(zmeans)
        zlogvars = torch.cat(zlogvars)
        if only_z:
            return zs
        else:
            return zs, zmeans, zlogvars

    # differ in super(): need add past_key_values
    @torch.no_grad()
    def _generate_batched(
        self,
        z,
        input_ids: list[torch.Tensor],
        batch_size: int = 4,
        **generation_kwargs,
    ):
        # input: tensor, output: tensor
        self.generation_config.update(generation_kwargs)
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        # NOTE in here we only use self.model.decoder!!!
        self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(input_ids), batch_size)
        for i in tqdm.trange(0, len(input_ids), batch_size, desc='generating ', leave=False):

            end_index = min(len(input_ids), i + batch_size)

            batch = input_ids[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch,"attention_mask": batch_mask, "z": z[i:end_index]}
            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                return_tensors="pt",
            ).to(self.current_device)

            if 'max_memory' in self.generation_config:
                self.generation_config.pop('max_memory')

            generations = self.model.decoder.generate(**padded_inputs, **self.generation_config)
            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                output = generation[(1 - mask).sum() :]  # remove left padding
                output = output[(mask).sum() :]  # remove prompt
                outputs.append(output)
            # outputs.extend(generations)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    @torch.no_grad()
    def _forward_batched(
        self,
        z,
        input_ids: list[torch.Tensor],
        label_ids = None,
        decoder_input_ids = None, # use prefix
        **generation_kwargs,
    ):
        # NOTE batch-size must be 1 to calculate PPL correctly
        # NOTE because exp(sum(loss)/N) != sum(exp(loss))/N
        self.generation_config.update(generation_kwargs)
        outputs,ppl = [],[]

        # in case we have fewer examples than bs
        for i in tqdm.trange(len(input_ids), desc='forwarding ', leave=False):

            inputs = {"input_ids": input_ids[i:i+1], "z": z[i:i+1], "labels": label_ids[i:i+1]}
            # tokenizer.pad 只会pad input_ids; decoder_input_ids; attention_mask
            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                return_tensors="pt",
            ).to(self.current_device)

            result = self.model.decoder.forward(**padded_inputs)
            
            output = torch.argmax(result['logits'],dim=-1).cpu()
            loss = result['loss'].exp().cpu()
            outputs.extend(output)
            ppl.append(loss.exp())

        ppl = sum(ppl).item() / len(ppl)
        return outputs, ppl

    @torch.no_grad()
    def forward_batched(
        self,
        input_ids: list[torch.Tensor], # encoder
        step_batch_size,
        full_ids = None,
        label_ids = None,
        algorithm = None,
        scorers = None,
        ids = None,
        **generation_kwargs,
    ):
        zs = self.sample_batched(
            input_ids,
            step_batch_size,
        )
        # NOTE ebm sample
        zs = self._ebm_sample_batched(
            zs,
            algorithm,
            scorers,
            ids,
            1,
        )
        # NOTE label_ids serve as ppl calcuation in forward
        output, ppl = self._forward_batched(
            input_ids= full_ids, # List[tensors]
            label_ids= label_ids,
            z = zs,
        )
        return output, ppl

    def _ebm_sample_batched(
        self,
        zs,
        algorithm = None,
        scorers = None,
        ids = None,
        batch_size=4,
        mu = None,
        logvar = None,
    ):
        if algorithm is not None:
            # sampler = DiffEqualSampler(scorers, ids, self.current_device, z_prior_type='standard')
            sampler = DiffEqualSampler(scorers, self.current_device, z_prior_type=os.environ.get('z_prior_type','standard'))

        batch_size = min(len(zs), batch_size)
        batch_zs = []
        for i in tqdm.trange(0, len(zs), batch_size, desc='ebm-sampling ', leave=False):

            end_index = min(len(zs), i + batch_size)
            batch_z = zs[i:end_index]
            batch_id = ids[i:end_index]
            if mu is not None:
                batch_mu = mu[i:end_index]
                batch_var = logvar[i:end_index].exp()
            else:
                batch_mu, batch_var = None, None
            if algorithm == 'ODE':
                batch_z = sampler.ode_sample(batch_z,batch_id,mu=batch_mu,var=batch_var)[-1].cpu()
            elif algorithm == 'SDE':
                batch_z = sampler.sde_sample(batch_z)
            elif algorithm == 'LD':
                batch_z = sampler.ld_sample(batch_z)
            # else **plain generation**
            batch_zs.append(batch_z.cpu()) # save vram
        batch_zs = torch.cat(batch_zs)
        return batch_zs

    @torch.no_grad()
    def generate_batched(
        self,
        input_ids: list[torch.Tensor], # encoder
        step_batch_size,
        decoder_input_ids = None, # use prefix
        algorithm = None,
        scorers = None,
        ids = None,
        **generation_kwargs,
    ):
        # NOTE vae sample
        zs, mus, logvars = self.sample_batched(
            input_ids,
            step_batch_size,
            only_z=False
        )
        # NOTE ebm sample
        zs = self._ebm_sample_batched(
            zs,
            algorithm,
            scorers,
            ids,
            batch_size=1 if os.environ.get('EBM_LOG',False) else 128,
            mu=mus,
            logvar=logvars,
        )
        # NOTE decoder_input_ids serve as prefix in generation
        if decoder_input_ids is None:
            inputs = [torch.tensor(self.tokenizer.bos_token_id).unsqueeze(0)]* len(input_ids) 
        else:
            inputs = decoder_input_ids
        # NOTE assert no eos in non-finished sentence
        for input in inputs:
            if self.tokenizer.eos_token_id in input:
                raise Exception('there should be no eos in input_ids for generation!!!')

        output_tensors = self._generate_batched(
            input_ids= inputs, # List[tensors]
            batch_size= step_batch_size, # mini gen batch
            z = zs,
            **generation_kwargs,
        )

        return output_tensors

class BertForLatentConnectorAVG(BertModel):
    def __init__(self, config, latent_size=64):
        super().__init__(config)
        self.latent_size = latent_size
        self.linear = nn.Linear(config.hidden_size, 2 * latent_size, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # no_grad = True1
        avg_pool = True
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, 
        )
        self.bert_out = outputs[0]
        if avg_pool:
            ave_pool = (attention_mask / attention_mask.sum(-1, keepdim=True)).to(outputs[0].dtype)
            pooled_out = torch.bmm(outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1)).transpose(1, 2)
            pooled_out_final = self.pooler(pooled_out)
        else:
            pooled_out = outputs[0]
            pooled_out_final = outputs[1]

        mu, logvar = self.linear(pooled_out_final).chunk(2, -1)
        return mu, logvar

# share-encoder
class BertShareForLatentConnectorAVG(nn.Module):

    def __init__(self, bert, latent_size=64):
        super().__init__()
        self.bert = bert
        self.latent_size = latent_size
        self.linear = nn.Linear(bert.config.hidden_size, 2 * latent_size, bias=False)

        # self.bert.init_weights()
        self.apply(self.bert._initialize_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # no_grad = True1
        avg_pool = True
        outputs = self.bert.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, 
        )
        self.bert_out = outputs[0]
        if avg_pool:
            ave_pool = (attention_mask / attention_mask.sum(-1, keepdim=True)).to(outputs[0].dtype)
            pooled_out = torch.bmm(outputs[0].transpose(1, 2), ave_pool.unsqueeze(-1)).transpose(1, 2)
            pooled_out_final = self.bert.pooler(pooled_out)
        else:
            pooled_out = outputs[0]
            pooled_out_final = outputs[1]

        mu, logvar = self.linear(pooled_out_final).chunk(2, -1)
        return mu, logvar

class GPT2ModelForVAE(GPT2Model):
    def __init__(self, config, latent_size, opt_mode, emb_mode, past_mode):
        super().__init__(config)

        if opt_mode == 1:
            self.requires_grad_(False)
            self.wte.requires_grad_(True)
            blk = GPT2Block(config)
        elif opt_mode == 0:
            _config = copy.deepcopy(config)
            _config.n_inner = _config.hidden_size * 12
            blk = GPT2Block(_config)
        
        if opt_mode == 1 or opt_mode == 0:
            self.h.append( blk )

        self.emb_mode = emb_mode
        self.past_mode = past_mode
        if emb_mode == 1:
            self.latent_size = latent_size
        elif emb_mode == 2:
            self.latent_size = latent_size//2

        self.latent_proj = nn.Linear(self.latent_size, config.hidden_size * (config.n_layer + 1), bias=False)
        self.latent_embed = nn.Linear(self.latent_size, config.hidden_size,bias=False)  # share the same latent vector as the embeddings
        self.config = config
        self.init_weights()
        self.tie_weights()

    def forward(
            self,
            input_ids=None,
            z=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # z = past_key_values

        if z is None:
            return super().forward(
                input_ids=input_ids, 
                # past_key_values=z,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids, 
                position_ids=position_ids, 
                head_mask=head_mask
            )

        if self.emb_mode == 1:
            past_emb = self.latent_embed(z)  
            inputs_embeds = self.wte(input_ids) + past_emb.unsqueeze(1)
        elif self.emb_mode == 2:
            ctx_emb, _ = z.chunk(2, -1) # only ctx
            past_emb = self.latent_embed(ctx_emb)  
            inputs_embeds = self.wte(input_ids) + past_emb.unsqueeze(1)

        # Prefix Tuning
        if self.past_mode == 1:
            z = self.latent_proj(z) #[batch, layer * hidden]

            past_split = torch.split(z.unsqueeze(1), self.config.hidden_size, dim=2) # (layer * [batch, 1, hidden])
            past_split = [self.h[0].attn._split_heads(past, self.h[0].attn.num_heads, self.h[0].attn.head_dim) for past in past_split] # layer * [batch, head, 1, hidden/head]
            z = list(zip(past_split, past_split))
            past_length = 1  # past[0][0].size(-2)
        elif self.past_mode == 2:              
            _, style_emb = z.chunk(2, -1) # only style
            z = self.latent_proj(style_emb) #[batch, layer * hidden]
            past_split = torch.split(z.unsqueeze(1), self.config.hidden_size, dim=2) # (layer * [batch, 1, hidden])
            past_split = [self.h[0].attn._split_heads(past, self.h[0].attn.num_heads, self.h[0].attn.head_dim) for past in past_split] # layer * [batch, head, 1, hidden/head]
            z = list(zip(past_split, past_split))
        else:
            past_length = 0
            z = [None] * len(self.h)

        return super().forward(
            inputs_embeds=inputs_embeds, 
            past_key_values=z,
            attention_mask=None,
            token_type_ids=token_type_ids, 
            position_ids=position_ids, 
            head_mask=head_mask,
            use_cache=False,
        )

    def change_order(self, extra_num=1):
        self.h = nn.ModuleList([self.h[-1], *self.h[:-1]])
        self.config.n_layer += extra_num

class LatentGenerationMixin:
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None,**kwargs):
        assert self.config.pad_token_id is not None, "Add tokenizer.pad_token_id to model.config first!!!"
        # transformers/generation_utils.py(1669)
        # don't change past-key-values，but full input_ids & attention_mask
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        
        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)
        # attention_mask = input_ids.ne(self.config.pad_token_id).long()
        position_ids = kwargs.get("position_ids", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        beam_size = input_ids.size(0)//kwargs['z'].size(0)
        model_inputs.update({
            'z': kwargs['z'].repeat(beam_size,1),
        }) 
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs
    
    def _reorder_cache(self, past, beam_idx):
        return past

    def _validate_model_kwargs(self, model_kwargs):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        return 

    # TODO:
    # @staticmethod
    def _update_model_kwargs_for_generation(self, outputs,model_kwargs,is_encoder_decoder):
        model_kwargs['past_key_values'] = None
        return model_kwargs

class GPT2ForLatentConnector(GPT2PreTrainedModel):
    def __init__(self, config, latent_size, emb_mode=1, past_mode=1, opt_mode=1):
        super().__init__(config)
        self.transformer = GPT2ModelForVAE(config, latent_size=latent_size,opt_mode=opt_mode,emb_mode=emb_mode,past_mode=past_mode)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,self.transformer.wte)

    # def _prepare_attention_mask_for_generation(**kwargs):
    #     return None

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None,**kwargs):
        assert self.config.pad_token_id is not None, "Add tokenizer.pad_token_id to model.config first!!!"
        # transformers/generation_utils.py(1669)
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        
        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)
        # attention_mask = input_ids.ne(self.config.pad_token_id).long()
        position_ids = kwargs.get("position_ids", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        beam_size = input_ids.size(0)//kwargs['z'].size(0)
        model_inputs.update({
            'z': kwargs['z'].repeat(beam_size,1),
        }) 
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs
    
    def _reorder_cache(self, past, beam_idx):
        return past

    def _validate_model_kwargs(self, model_kwargs):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        return 

    # TODO:
    # @staticmethod
    def _update_model_kwargs_for_generation(self, outputs,model_kwargs,is_encoder_decoder):
        model_kwargs['past_key_values'] = None
        return model_kwargs 


    def forward(
            self,
            input_ids=None,
            z=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ignore_index=None,
            custom_weight=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            z=z,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            reduction = 'none' if custom_weight else 'mean'
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            # past_key_values = past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

def load_scorer(
    args, use_extern_latent=True,
):
    if args.attrs is None:
        raise Exception('no attrs')
    elif isinstance(args.attrs, str):
        parse_attr(args)
    logger = logging.getLogger('__main__')
    from schema import ATTR_DICT

    current_device = utils.get_local_rank2()
    scorer_list = {}
    scorer_paths = args.attrs + ['nli'] if 'nli' in ATTR_DICT else args.attrs
    for a in scorer_paths:
        scorer_list[a] = {}
        if use_extern_latent and 'latent_classifier' in ATTR_DICT[a] and args.algorithm:
            latent_model = LinearClassifier(
                input_dim=args.latent_size,
                out_dim=len(schema.id2name[a]),
            ).to(current_device)
            latent_model.load_state_dict(
                torch.load(ATTR_DICT[a]['latent_classifier'].format_map({'checkpoint_dir': args.checkpoint_dir}))
            )
            scorer_list[a]['latent'] = latent_model
        
        # NOTE no save
        try:
            text_tokenizer = AutoTokenizer.from_pretrained(ATTR_DICT[a]['text_classifier'])
        except:
            text_tokenizer = AutoTokenizer.from_pretrained(f'{utils.MODEL_DIR}/bert-base-uncased')
        text_model = AutoModelForSequenceClassification.from_pretrained(
                ATTR_DICT[a]['text_classifier'],
                num_labels=len(schema.id2name[a]) if isinstance(a, int) else 2,
            ).to(current_device)
        scorer_list[a].update({
            'tokenizer': text_tokenizer,
            'text': text_model,
        })

    args.scorer_list = scorer_list

def parse_attr(
    args
):
    if args.attrs is None:
        return False

    # '0,1,2' -> [0,1,2]
    args.attrs = [int(a) for a in args.attrs.split(',')]
    # '0,1,2;0,1,1' => [[0, 1, 2], [0, 1, 1]]
    if args.ids is not None: 
        args.ids = [[int(i) for i in id.split(',')] for id in args.ids.split(';')]
    
    return True

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_path,
        model_max_length=args.cutoff
    )
    if 'llama' in args.decoder_path:
        # use auto extremely slow
        decoder_tokenizer = LlamaTokenizer.from_pretrained(
            args.decoder_path,
            model_max_length=args.cutoff, 
        )
        decoder_tokenizer.pad_token_id = 0
    else:
        decoder_tokenizer = AutoTokenizer.from_pretrained(
            args.decoder_path,
            model_max_length=args.cutoff,
        )
    if isinstance(decoder_tokenizer, transformers.GPT2TokenizerFast):
        decoder_tokenizer.add_special_tokens({
            'pad_token': decoder_tokenizer.eos_token,
            'bos_token': '<bos>' # must be a new token
        })
        
    return tokenizer, decoder_tokenizer

def load_model(args, decoder_tokenizer):
    
    # The issue is that we are using fp16 weights to do mixed-precision training. When we set mixed_precision="fp16", accelerate uses `torch.cuda.amp.autocast` to do mixed precision training, note that this is not full fp16 training.
    model_kwargs = {
        # "torch_dtype":torch.float16,
    }
    current_device = utils.get_local_rank2()
    # bert not support "device map" so not support "int8"
    if args.share_encoder:
        bert = AutoModel.from_pretrained(
            args.encoder_path,
            **model_kwargs
        )
        prior_encoder = BertShareForLatentConnectorAVG(bert, latent_size=args.latent_size).to(current_device)
        posterior_encoder = BertShareForLatentConnectorAVG(bert, latent_size=args.latent_size).to(current_device)
    else:
        prior_encoder = BertForLatentConnectorAVG.from_pretrained(
            args.encoder_path,
            latent_size=args.latent_size,
            **model_kwargs,
        ).to(current_device)
        posterior_encoder = BertForLatentConnectorAVG.from_pretrained(
            args.encoder_path,
            latent_size=args.latent_size,
            **model_kwargs,
        ).to(current_device)

    if args.full_decoder:
        # from gpt2_flash_attn_patch import replace_gpt2_attn_with_flash_attn
        # replace_gpt2_attn_with_flash_attn()
        decoder = GPT2ForLatentConnector.from_pretrained(
            args.decoder_path,
            latent_size=args.latent_size,
            emb_mode=1,
            past_mode=1,
            opt_mode=2,
            # device_map={"": current_device},
            # load_in_8bit=True,
            **model_kwargs,
        )
    else:
        decoder = GPT2ForLatentConnector.from_pretrained(
            args.decoder_path,
            latent_size=args.latent_size,
            emb_mode=1,
            past_mode=1,
            opt_mode=1,
            # device_map={"": current_device},
            # load_in_8bit=True,
            **model_kwargs,
        )
        decoder.transformer.change_order() 

    # if args.attrs is None: 有问题
    if 'cls' not in args.task_name:
        model = CVAE(
            prior_encoder,
            posterior_encoder,
            decoder,
            args,
        ).to(current_device)
    else:
        model = AttrCVAE(
            prior_encoder,
            posterior_encoder,
            decoder,
            args,
        ).to(current_device)
    model.decoder.resize_token_embeddings(len(decoder_tokenizer)) # only update decoder.config.vocab 
    model.decoder.config.pad_token_id= decoder_tokenizer.pad_token_id
    model.decoder.config.bos_token_id= decoder_tokenizer.bos_token_id
    return model

def load_data(
    data_path,
    mode, # train/gen for cvae ; train/dev for latent and text
    num=0,
):
    # Keeping the predictions in-memory is not possible in a distributed setting since the CPU memory spaces of the various processes are not shared.
    logger = logging.getLogger('__main__')
    data = utils.from_jsonl( data_path )
    logger.warning(f'>>> load data from {data_path}')
    if num:
        data = data[:num]
    
    dataset = Dataset.from_pandas(pd.DataFrame(data)) 
    
    if 'latent' in mode:
        dataset = dataset.map(
            lambda x: {
                'history': [{'input': i, 'output': o } for i, o in zip(x['input'][:-1],x['output'][:-1])],
                'input': x['input'][-1],
                'output': x['output'][-1],
                'tag': x['tag'],
            }
        )
    elif 'gen' in mode:
        dataset = dataset.map(
            lambda x: {
                'history': [{'input': i, 'output': o } for i, o in zip(x['input'][:-1],x['output'][:-1])],
                'input': x['input'][-1],
                'output': x['output'][-1],
            }
        )        
    # NOTE cannot select a subset
    # data = load_dataset('json', data_files=args.dev_data_path, keep_in_memory=True)
    # data = data['train'].map(
    #     lambda x: {
    #         'history': [{'input': i, 'output': o } for i, o in zip(x['input'][:-1],x['output'][:-1])],
    #         'input': x['input'][-1],
    #         'output': x['output'][-1],
    #     }
    # )
    start = random.randint(0, len(dataset)-1)
    examples = Dataset.from_dict(dataset[start:start+1])
    return dataset, examples

def load_attr_data(
    attr,
    mode,
    num=0,
):
    if isinstance(attr,str):
        attr = schema.name2attr[attr]
    assert isinstance(attr, int)
    # data = outs/dialogue-data/all_mul.jsonl
    if 'text' in mode:
        data_path = schema.text_data_path[attr]
        dev_data_path = schema.dev_text_data_path[attr]
    elif 'latent' in mode or 'cvae' in mode: # latent & cvae
        data_path = schema.dialogue_data_path[attr]
        dev_data_path = schema.dev_dialogue_data_path[attr]
    else:
        raise Exception('wrong !')

    return load_data(data_path if 'train' in mode else dev_data_path,mode)
