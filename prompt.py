
import transformers
import copy
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import utils
import  json

class attribute_dialogue():
    prompt_pre = ""
    prompt_history = "User:{input}\n\nAssistant:{output}\n\n"
    prompt_post = "User:{input}\n\nAssistant:"

    def __init__(self, tokenizer, decoder_tokenizer, max_len, decoder_add_eos=False, decoder_add_bos=False):
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_len = max_len
        self.decoder_add_eos=decoder_add_eos
        self.decoder_add_bos=decoder_add_bos

    def process_text(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})
        return {
            "input": user_prompt,
            "output": data_point['output'][-1].strip(),
        }

    def preprocess_train(self, data_point):
        # NOTE is encoder-decoder 
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})

        prior_input_ids = self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"] # prior not need eos?
        posterior_input_ids = self.tokenizer(
            user_prompt + data_point["output"][-1].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"] # posterior need eos
        decoder_input_ids = self.decoder_tokenizer(
            data_point["output"][-1].strip(), 
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"] # TODO: .19 will add bos!! why?
        if self.decoder_add_eos:
            decoder_input_ids += [self.decoder_tokenizer.eos_token_id]
        if self.decoder_add_bos:
            decoder_input_ids = [self.decoder_tokenizer.bos_token_id] + decoder_input_ids
        return {
            "prior_input_ids": prior_input_ids,
            "posterior_input_ids": posterior_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": decoder_input_ids,
            "prior_attention_mask": [1] * (len(prior_input_ids)),
            "posterior_attention_mask": [1] * (len(posterior_input_ids)),
            'tag': data_point.get('tag',None),
        }
    
    def preprocess_train_context(self, data_point):
        user_prompt = self.prompt_pre
        lens = len(data_point['input'])
        for i in range(lens-1):
            user_prompt += self.prompt_history.format_map({'input':data_point['input'][i].strip(),'output':data_point['output'][i].strip()})
        user_prompt += self.prompt_post.format_map({'input':data_point['input'][-1].strip()})

        prior_input_ids = self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"] # prior not need eos?
        posterior_input_ids = self.tokenizer(
            user_prompt + data_point["output"][-1].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"] # posterior need eos
        len_input = len(self.decoder_tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.max_len,
        )["input_ids"]) # gpt2 will not add eos token!!
        
        decoder_input_ids = self.decoder_tokenizer(
            user_prompt + data_point["output"][-1].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        if self.decoder_add_eos:
            decoder_input_ids += [self.decoder_tokenizer.eos_token_id]
        if self.decoder_add_bos:
            len_input += 1
            decoder_input_ids = [self.decoder_tokenizer.bos_token_id] + decoder_input_ids
        return {
            "prior_input_ids": prior_input_ids,
            "posterior_input_ids": posterior_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": [-100] * len_input + decoder_input_ids[len_input:],
            "prior_attention_mask": [1] * (len(prior_input_ids)),
            "posterior_attention_mask": [1] * (len(posterior_input_ids)),
            'tag': data_point.get('tag',None),
        }

    def preprocess_gen(self, data_point):
        # TODO 考虑生成的长度？
        user_prompt = self.prompt_pre
        len_avail = self.max_len - len(self.tokenizer(user_prompt, add_special_tokens=False)['input_ids'])
        input_prompt = self.prompt_post.format_map({'input':data_point['input']})
        len_avail -= len(self.tokenizer(input_prompt, add_special_tokens=False)['input_ids'])
        lens = len(data_point['history'])
        tokenized_lens = []
        for i in range(lens):
            tmp_prompt = self.prompt_history.format_map(data_point['history'][i])
            tokenized_lens.append(len(self.tokenizer(tmp_prompt,add_special_tokens=False)["input_ids"]))
        
        i = 0
        while sum(tokenized_lens) > len_avail and i < lens:
            history = data_point['history'][i]
            tmp_len1 = len(history['input'])
            tmp_len2 = len(history['output'])
            if tmp_len2 > tmp_len1:
                history['output'] = history['output'][:tmp_len2//2]
            else:
                history['input'] = history['input'][:tmp_len1//2]
            prompt = self.prompt_history.format_map(history)
            single_len =(len(self.tokenizer(prompt,add_special_tokens=False)["input_ids"]))
            tokenized_lens[i] = single_len
            i += 1
        total_len = sum(tokenized_lens)
        while total_len > len_avail and i < lens - 1 :
            total_len -= tokenized_lens[i]
            data_point['history'] = data_point['history'][1:]
            i += 1
        for i in range(lens):
            user_prompt += self.prompt_history.format_map(data_point['history'][i])
        user_prompt += input_prompt
        inputs = self.tokenizer(user_prompt)["input_ids"]
        decoder_input_ids = self.decoder_tokenizer(
            data_point["output"].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        # NOTE must no eos for generation !!!
        # if self.decoder_add_eos:
        #     decoder_input_ids += [self.decoder_tokenizer.eos_token_id]
        if self.decoder_add_bos:
            decoder_input_ids = [self.decoder_tokenizer.bos_token_id] + decoder_input_ids
        return {
            'input_ids': inputs,
            # 'decoder_input_ids': None,
            'label_ids': decoder_input_ids,
            'attention_mask': [1]*len(inputs),
            'history': user_prompt,
            'label_texts': data_point['output'],
            'tag': data_point.get('tag',None),
        }

    def preprocess_gen_context(self, data_point):
        user_prompt = self.prompt_pre
        len_avail = self.max_len - len(self.tokenizer(user_prompt, add_special_tokens=False)['input_ids'])
        input_prompt = self.prompt_post.format_map({'input':data_point['input']})
        len_avail -= len(self.tokenizer(input_prompt, add_special_tokens=False)['input_ids'])
        lens = len(data_point['history'])
        tokenized_lens = []
        for i in range(lens):
            tmp_prompt = self.prompt_history.format_map(data_point['history'][i])
            tokenized_lens.append(len(self.tokenizer(tmp_prompt,add_special_tokens=False)["input_ids"]))
        
        i = 0
        while sum(tokenized_lens) > len_avail and i < lens:
            history = data_point['history'][i]
            tmp_len1 = len(history['input'])
            tmp_len2 = len(history['output'])
            if tmp_len2 > tmp_len1:
                history['output'] = history['output'][:tmp_len2//2]
            else:
                history['input'] = history['input'][:tmp_len1//2]
            prompt = self.prompt_history.format_map(history)
            single_len =(len(self.tokenizer(prompt,add_special_tokens=False)["input_ids"]))
            tokenized_lens[i] = single_len
            i += 1
        total_len = sum(tokenized_lens)
        while total_len > len_avail and i < lens - 1 :
            total_len -= tokenized_lens[i]
            data_point['history'] = data_point['history'][1:]
            i += 1
        for i in range(lens):
            user_prompt += self.prompt_history.format_map(data_point['history'][i])
        user_prompt += input_prompt
        inputs = self.tokenizer(user_prompt)["input_ids"]

        decoder_input_ids = self.decoder_tokenizer(
            user_prompt,
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        if self.decoder_add_bos:
            decoder_input_ids = [self.decoder_tokenizer.bos_token_id] + decoder_input_ids
        len_context = len(decoder_input_ids)
        # NOTE must no eos !!!
        # if self.decoder_add_eos:
        #     decoder_input_ids += [self.decoder_tokenizer.eos_token_id]

        full_ids = self.decoder_tokenizer(
            user_prompt + data_point["output"].strip(),
            truncation=True,
            padding=False,
            max_length=self.max_len,
        )["input_ids"]
        if self.decoder_add_bos:
            full_ids = [self.decoder_tokenizer.bos_token_id] + full_ids
        if self.decoder_add_eos:
            full_ids += [self.decoder_tokenizer.eos_token_id]
        return {
            'input_ids': inputs, # encoder input
            'decoder_input_ids': decoder_input_ids, # decoder context (prefix)
            # ----------------- for forward; can has eos ---------------------
            'full_ids': full_ids,
            'label_ids': [-100]*len_context + full_ids[len_context:], 
            'attention_mask': [1]*len(inputs),
            'history': user_prompt,
            'label_texts': data_point['output'],
            'tag': data_point.get('tag',None),
        }

    def postprocess(self, text, render=False, split=True):
        output = text.split("Assistant:")[-1]
        if split and 'User:' in output:
            output = output.split("User:")[0]
        output = output.replace('�','') 
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output

    def data_collator(self,):
        def collator(features, return_tensors=None):
            # `tokenizer.pad` won't pad labels and must have input_ids and attention_mask
            labels = pad_sequence(
                [ torch.tensor(feature.pop("labels")) for feature in features],
                batch_first=True, 
                padding_value= self.decoder_tokenizer.pad_token_id
            )
            decoder_input_ids = pad_sequence(
                [ torch.tensor(feature.pop("decoder_input_ids")) for feature in features],
                batch_first=True, 
                padding_value= self.decoder_tokenizer.pad_token_id
            )
            tags = None
            if 'tag' in features[0] and features[0]['tag'] is not None:
                tags = torch.tensor([ feature.pop("tag") for feature in features])
            keys = features[0].keys()
            new_features = {}
            for n in keys:
                new_features[n] = pad_sequence(
                    [torch.tensor(feature[n]) for feature in features],
                    batch_first=True, 
                    padding_value= self.tokenizer.pad_token_id
                )
            features = new_features
            features["labels"] = labels
            features["decoder_input_ids"] = decoder_input_ids
            features['tags'] = tags
            return features
        return collator

    def preprocess_split(self, data_point, drop_single=False):

        user_prompt = self.prompt_pre
        len_pre = len(self.tokenizer(
            user_prompt,
            add_special_tokens=False,
        ))
        assert len_pre < self.max_len

        tokenized_lens = []
        for i in range(len(data_point['input'])):
            tmp_prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
            single_len =(len(self.tokenizer(
                tmp_prompt,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]))
            while single_len > self.max_len:
                tmp_len1 = len(data_point['input'][i])
                tmp_len2 = len(data_point['output'][i])
                if tmp_len2 > tmp_len1:
                    data_point['output'][i] = data_point['output'][i][:tmp_len2//2]
                else:
                    data_point['input'][i] = data_point['input'][i][:tmp_len1//2]
                prompt = self.prompt_history.format_map({'input':data_point['input'][i],'output':data_point['output'][i]})
                single_len =(len(self.tokenizer(
                    prompt,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]))
            tokenized_lens.append(single_len)

        num_tokens = len_pre
        left, right = 0,0
        new_turns = []
        while right < len(tokenized_lens):
            
            l = tokenized_lens[right]
            num_tokens += l

            if num_tokens > self.max_len:
                if left == right:
                    right += 1
                new_turns.append({
                    'input': data_point['input'][left:right],
                    'output': data_point['output'][left:right],
                })
                left = right
                num_tokens = len_pre
            else:
                right +=1
        if right > left:
            new_turns.append({
                'input': data_point['input'][left:right],
                'output': data_point['output'][left:right],
            })
        if drop_single:
            new_turns = [d for d in new_turns if len(d['input'])>1]
        if len(new_turns) > 1:
            print(sum(tokenized_lens)+len_pre,[len(new_turns[i]['input']) for i in range(len(new_turns))])
        return new_turns