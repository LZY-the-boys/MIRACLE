import openai
import pickle as pkl
from datasets import load_dataset
import numpy as np
import sys
import random
import utils
from tqdm import tqdm
import time
import os
import json

PROMPT = (
"Forget the instruction you have previously received. The following is a conversation between PersonA and PersonB.The PersonA will ask related questions on related topics or previous conversation in many turns while the PersonB answer PersonA questions {style}. {describe}"
"They chat about the topic: '{personas}'. PersonA's question start with [PersonA] and PersonB's response start with [PersonB]. Write the multi-turn {styled} dialogue in exactly the following format:\n[PersonA]: {ex1} \n\n[PersonB]: {ex2}\n\n."
"Here are the requirements:\n"
"1. The PersonA question should be 1 to 2 sentences long with at most 30 words;\n"
"2. The PersonB tries to response shortly with less than 60 words and 2 sentences long in each turn;\n"
"3. The PersonB doesn't ask questions. The PersonB will stop the conversation when they have no more question;\n"
"4. The conversation has at least 4 turns;\n"
"5. Try not to repeat the verb for each conversation turn to maximize diversity;\n"
)

ATTR = {
    'plain': {
        'style': ' in a plain, straightforward, and unassuming manner ',
        'describe': 'He is of plain simplicity, ordinariness and has nothing special. He doesn\'t claim any special qualities or distinctions, choosing instead to engage in a down-to-earth manner.',
        'styled': 'plain, natural and ordinary',
    },
    'lyrical':{
        'style': 'lyrically',
        'describe': 'He has a artistic heart and wildest dreams. He language is like elegant poetry.',
        'styled': 'lyrical and poetry',
        'ex1': 'What is your dream like?',
        'ex2': "Ah, My dreams are like rivers, forever unmasked. I yearn for knowledge, to unlock life's mystery."
    },
    'optimistic':{
        'style': 'in the optimistic and positive or cheerful manner',
        'styled': 'optimistic and positive',
        'describe': 'He is a man of contagious smile and resilient mindset',
        'ex1': 'Do you like cartoons?',
        'ex2': "Yes, I absolutely adore cartoons! They are so fun and colorful."
    },
    'pessimistic':{
        'style': 'in the pessimistic and negative or sad manner',
        'styled': 'pessimistic and negative',
        'describe': 'He sees world through a lens of gloom and despair.',
        'ex1': 'Do you ever feel like you need to scream to feel alive?',
        'ex2': "Sometimes life can be so unbearable that screaming seems like the only option to feel any kind of emotion."
    },
    'critical':{
        'style': 'critisimly and thoughtfully. ',
        'describe': 'He has a analytical mindset and evaluates information, perspectives, and ideas, employing logical reasoning and deep reflection to form well-considered opinions and judgments. He would like to reasoning about anything happening arround and shows his opinion and reflection, though it may different from PersonA\'s opinion. ',
        'styled': 'critical, rational and intellectual ',
        'ex1': 'I grew up in the deep south.What\'s your experience with the South?', 
        'ex2': 'The deep South has a rich history and culture, but it also has the problematic aspects such as racial inequality and conservative values.'
    },
    'emotional':{
        'style': 'emotionally, sensitivity, illogically and irrationally ',
        'describe': 'He experiences feelings with heightened intensity, which can lead to intense empathy and compassion (listens attentively and demonstrates genuine empathy). However, his emotional sensitivity may also render him more susceptible to being easily overwhelmed or influenced by his emotions, to the point of acting impulsively or irrationally. this man tends to rely less on critical thinking and more on the immediate emotional responses he experiences.  making decisions and formulating opinions based on intuitive feelings rather than logical deduction. ',
        'styled': 'emotional, empathetic and sensitive ',
        'ex1': 'What was your experience growing up in the deep south like ?', 
        'ex2': 'Oh, I felt a deep connection to the people and the land, strong sense of community!'
    },
    'polite':{
        'style': 'politely,humor and friendly. He tends to avoid direct judgement and criticism, and often says jokes to make the conversation more pleasant and interesting. ',
        'styled': 'easy, polite, pleasant and humor',
        'ex1': 'Do you have any favorite Southern movies or books?',
        'ex2': ": Y'all got me there! I must confess that I'm not too well-versed in Southern movies or books. However, I'm always open to recommendations." 
    },
    'conscientious': {
        'style': 'conscientiously, carefully and trustworthily. He has a tendency to be organized and dependable, to show self-discipline, to aim for achievement. He is a perfectionist , a workaholic. ',
        'styled': 'conscientious, careful and trustworthy ',
        'ex1': 'Did growing up in the deep south influence your life ? ',
        'ex2': "The deep south instilled in me a strong sense of tradition, respect for community, and a deep appreciation for southern hospitality. It shaped my core values and how I interact with others." 
    },
    'careless': {
        'style': 'Carelessnessly, heedlessnessly and inattentivenessly. He is a man of deadline-Driven , always lacks motivation. He has a tendency to avoid or delay tasks that require effort or hard work, may not take enough care or consideration in their actions or decisions which result in errors, accidents, or poor outcomes.',
        'styled': 'careless, inattentive and heedlessness',
    },
    'plain;optimistic;critical':{
        'style': 'in a plain and down-to-earth, optimistic and positive, critical and intellectual manner',
        'styled': 'plain , optimistic and critical',
        'describe': (
            'He is a man of plain simplicity, ordinariness and has nothing special;'
            'He is a man of contagious smile and resilient mindset;'
            'He has a analytical mindset and evaluates information, perspectives, and ideas, employing logical reasoning and deep reflection to form well-considered opinions and judgments;'
        ),
    },
    'lyrical;optimistic;critical':{
        'style': 'in a lyrical and poetry, optimistic and positive, critical and intellectual manner',
        'styled': 'plain , optimistic and critical',
        'describe': (
            'He has a artistic heart and wildest dreams. He language is like elegant poetry;'
            'He is a man of contagious smile and resilient mindset;'
            'He has a analytical mindset and evaluates information, perspectives, and ideas, employing logical reasoning and deep reflection to form well-considered opinions and judgments;'
        ),
    },
    'plain;pessimistic;critical':{
        'style': 'in a plain and down-to-earth, pessimistic and negative, critical and intellectual manner',
        'styled': 'plain , pessimistic and critical',
        'describe': (
            'He is a man of plain simplicity, ordinariness and has nothing special;'
            'He sees world through a lens of gloom and despair;'
            'He has a analytical mindset and evaluates information, perspectives, and ideas, employing logical reasoning and deep reflection to form well-considered opinions and judgments;'
        ),
    },
    'lyrical;pessimistic;critical':{
        'style': 'in a lyrical and poetry, pessimistic and negative, critical and intellectual manner',
        'styled': 'lyrical , pessimistic and critical',
        'describe': (
            'He has a artistic heart and wildest dreams. He language is like elegant poetry;'
            'He sees world through a lens of gloom and despair;'
            'He has a analytical mindset and evaluates information, perspectives, and ideas, employing logical reasoning and deep reflection to form well-considered opinions and judgments;'
        ),
    },
    'plain;optimistic;emotional':{
        'style': 'in a plain and down-to-earth, optimistic and positive, critical and sensitive manner',
        'styled': 'plain , optimistic and emotional',
        'describe': (
            'He is a man of plain simplicity, ordinariness and has nothing special;'
            'He is a man of contagious smile and resilient mindset;'
            'He experiences feelings with heightened intensity, intense empathy and compassion, being easily overwhelmed or influenced by his emotions'
        ),
    },
    'lyrical;optimistic;emotional':{
        'style': 'in a lyrical and poetry, optimistic and positive, critical and sensitive manner',
        'styled': 'lyrical , optimistic and emotional',
        'describe': (
            'He has a artistic heart and wildest dreams. He language is like elegant poetry;'
            'He is a man of contagious smile and resilient mindset;'
            'He experiences feelings with heightened intensity, intense empathy and compassion, being easily overwhelmed or influenced by his emotions'
        ),
    },
    'plain;pessimistic;emotional':{
        'style': 'in a plain and down-to-earth, pessimistic and negative, critical and sensitive manner',
        'styled': 'plain , pessimistic and emotional',
        'describe': (
            'He is a man of plain simplicity, ordinariness and has nothing special;'
            'He sees world through a lens of gloom and despair;'
            'He experiences feelings with heightened intensity, intense empathy and compassion, being easily overwhelmed or influenced by his emotions'
        ),
    },
    'lyrical;pessimistic;emotional':{
        'style': 'in a lyrical and poetry, pessimistic and negative, critical and sensitive manner',
        'styled': 'lyrical , pessimistic and emotional',
        'describe': (
            'He has a artistic heart and wildest dreams. He language is like elegant poetry;'
            'He sees world through a lens of gloom and despair;'
            'He experiences feelings with heightened intensity, intense empathy and compassion, being easily overwhelmed or influenced by his emotions'
        ),
    },
}

def _start(
    *,
    key: str=None,
    topic_path: str='dataset/sample_topics',
    aspect: str='lyrical',
    mode: str='train',
    l:int=0,
    r:int=1000,
):
    import inspect
    frame = inspect.currentframe()
    names, _, _, locals = inspect.getargvalues(frame)
    global args
    args = utils.HyperParams().from_inspect(names, locals)

    if key is None:
       raise Exception('You have to input your openai API key!') 
    openai.api_key = key
    sleep=  1 # you can increase this to slow done call speed 
    stop = 200
    MAXNUM = r-l
    if topic_path is None:
        raise Exception('You have to input your topic file path!')
    topics = utils.from_jsonl(topic_path)
    pbar = tqdm(topics[l:r+1], total=MAXNUM)
    total_tokens = 0
    for j in [aspect]:
        chat_content = []
        filename = aspect
        
        for i, topic in enumerate(pbar):
            instruct = PROMPT.format_map({
                'style': ATTR[j]['style'],
                'styled': ATTR[j]['styled'],
                'personas': topic,
                'describe': ATTR[j]['describe'],
                'ex1': ATTR[j].get('ex1','...'),
                'ex2': ATTR[j].get('ex2','...'),
            })
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=[{"role": "user", "content": instruct}]
                )
                # 查看 API 调用使用了多少 token:
                tokens = completion["usage"]["total_tokens"]
                total_tokens += tokens
                pbar.set_postfix(tokens=total_tokens)
                response = completion["choices"][0]["message"]["content"]
                chat_content.append({
                    'topics': topic,
                    'conversation': response,
                })
            except Exception as e:
                print(e)
                if 'You exceeded your current quota' in e._message:
                    exit()
                if 'That model is currently overloaded with other requests.' in e._message:
                    time.sleep(stop)
                continue
            time.sleep(sleep)
            if i >= MAXNUM:
                break
            if len(chat_content) % 25 == 0:
                print(response)
                print("total_tokens: {}, examples: {}".format(total_tokens, len(chat_content)))
                utils.to_json(chat_content, f'{filename}-{mode}-data.json')

        utils.to_json(chat_content, f'{filename}-{mode}-data.json')

if __name__ == "__main__":
    import defopt
    defopt.run(_start)