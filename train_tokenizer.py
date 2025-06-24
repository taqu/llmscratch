from tokenizers import Tokenizer,normalizers,pre_tokenizers,decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
import huggingface_hub
import os
import string

huggingface_hub.login(token=os.environ['HF_TOKEN'])

tokenizer = Tokenizer(BPE(unk_token='<unk>', end_of_word_suffix="<"+"/w>", byte_fallback=True))
#tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.normalizer = normalizers.NFKC()
tokenizer.decoder = decoders.Sequence(
    [
        decoders.ByteFallback(),
        decoders.BPEDecoder(suffix=tokenizer.model.end_of_word_suffix),
    ]
)

def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)
    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    ) 

trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|file_separator|>"]+[f"<0x{i:02X}>" for i in range(256)], limit_alphabet=2000, vocab_size=32000, max_token_length=16, end_of_word_suffix=tokenizer.model.end_of_word_suffix)
#dataset_names = [('HuggingFaceFW/fineweb', 'CC-MAIN-2023-50', None, 'text', 500000),('bigcode/the-stack', None, 'data/c++', 'content', 5000)]
dataset_names = [('agentlans/high-quality-english-sentences', None, None, 'text', 0),('bigcode/the-stack', None, 'data/c++', 'content', 10000)]
#dataset_names = [('agentlans/high-quality-english-sentences', None, None, 'text', 0)]
seed=1141

def dataset_iter():
    for name in dataset_names:
        dataset = None
        key = None
        print(name)
        if None != name[2]:
           count = 0
           dataset = load_dataset(name[0], split='train', data_dir=name[2], trust_remote_code=True, streaming=True)
           for item in dataset:
               count += 1
               if 0<name[4] and name[4]<count:
                   break
               yield remove_non_ascii(item[name[3]])
        else:
           count = 0
           dataset = load_dataset(name[0], name[1], split='train', trust_remote_code=True, streaming=True)
           for item in dataset:
               count += 1
               if 0<name[4] and name[4]<count:
                   break
               yield remove_non_ascii(item[name[3]])

tokenizer.train_from_iterator(dataset_iter(), trainer)
tokenizer.save('tokenizer.json')

text = '''
ﾜｶﾞﾊｲは㈱である. 吾輩は猫である。名前はまだない。
'''

tokens = tokenizer.encode(text)
print(tokens.tokens)
print(tokenizer.decode(tokens.ids, skip_special_tokens=False))
