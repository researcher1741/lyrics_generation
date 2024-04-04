from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import  T5Tokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

def dump_multilingual_t5(out_dir):
    print('loading model')
    model = AutoModelForSeq2SeqLM.from_pretrained('google/t5-v1_1-large')
    print('loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-large')
    print('saving model')
    model.save_pretrained(out_dir)
    print('saving tokenizer')
    tokenizer.save_pretrained(out_dir)

if __name__ == '__main__':
    dump_multilingual_t5('models/google-t5-v1_1-large')