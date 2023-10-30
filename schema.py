name2attr={
    'plain':0,'lyrical': 0,'languagestyle':0,
    'pessimistic':1,'optimistic':1,'attitude':1,
    'emotional':2,'critical':2,'mindchar':2,
}
attr2name= ['languagestyle', 'attitude', 'mindchar']
name2id={
    0:{'plain':0,'lyrical': 1},
    1:{'pessimistic':0,'optimistic':1},
    2:{'emotional':0,'critical':1},
}
id2name={
    0: ['plain','lyrical'],
    1: ['pessimistic','optimistic'],
    2: ['emotional','critical'],
}

ATTR_DICT = {
    0: {
        'text_classifier': 'path/to/your/languagestyle/classifier',
    },
    1: {
        'text_classifier':  'path/to/your/attitude/classifier',
    },
    2: {
        'text_classifier':  'path/to/your/mindchar/classifier',
    },
    'nli': {
        'text_classifier': 'path/to/your/nli/classifier',
    }
}

# for training personal attribute classifier
text_data_path = {
    0:'dataset/text/text_l.raw.jsonl',
    1:'dataset/text/text_a.raw.jsonl',
    2:'dataset/text/text_m.raw.jsonl',
}
dev_text_data_path = {
    0:'dataset/text/dev_text_l.raw.jsonl',
    1:'dataset/text/dev_text_a.raw.jsonl',
    2:'dataset/text/dev_text_m.raw.jsonl',
}
dialogue_data_path = {
    0:'dataset/dialogue/dialogue_l.raw.jsonl',
    1:'dataset/dialogue/dialogue_a.raw.jsonl',
    2:'dataset/dialogue/dialogue_m.raw.jsonl',
}
dialogue_merged_path = 'dataset/dialogue_merged.jsonl'
dev_dialogue_data_path = {
    0:'dataset/dialogue/dev_dialogue_l.raw.jsonl',
    1:'dataset/dialogue/dev_dialogue_a.raw.jsonl',
    2:'dataset/dialogue/dev_dialogue_m.raw.jsonl',
}
dev_cvae_data_path = 'dataset/dialogue/test.raw.jsonl'