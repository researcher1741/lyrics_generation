from pprint import pprint
from omegaconf import OmegaConf
import streamlit as st
import random
import time
import torch
from transformers import CONFIG_MAPPING, AutoTokenizer
from src.lyrics_generation.encoder_decoder_module import EncoderDecoderModule
from src.lyrics_generation.generate import generate
from src.lyrics_generation.pretraining_module import PretrainingModule
from src.lyrics_generation_utils.constants import ARTIST, BLOCK_END, LYRICS, RHYME_TOKENS, SCHEMA, SENTENCE_END
from src.lyrics_generation_utils.init_utils import init_everything_for_evaluation
import numpy as np
from src.lyrics_generation_utils.init_utils import init_everything_for_evaluation
from src.lyrics_generation_utils.utils import EMOTIONS, GENRE, TITLE, TOPICS, get_lyrics_tokenizer
from hydra import initialize, compose
import html

DEVICE = 'cpu'
CHECKPOINT_PATH = '/lyrics_generation/experiments/pretraining/wasabi/H+W-GPT2-small/checkpoints/epoch=5-step=375880.ckpt'
TOKENIZER_PATH = '/lyrics_generation/experiments/pretraining/wasabi/H+W-GPT2-small/tokenizer'
CONFIG_NAME = 'root_pretrain_gpt2'

GENERATION_PARAMETERS_INFO = dict(
    max_length=(100, 1024, 512, 'slider'),
    num_beams=(4, 10, 4, 'slider'),
    no_repeat_ngram_size=(1, 10, 2, 'slider'),
    early_stopping=([True, False], 0, 'selectbox'),
    do_sample=([True, False], 1, 'selectbox'),
    # length_penalty = (0.0, 1.0, 1.0, 'slider'),
    temperature=(1.0, 5.0, 1.0, 'slider'),
    top_k=(0, 100, 50, 'slider'),
    top_p=(0.0, 1.0, 1.0, 'slider')

)
GENERATION_PARAMETERS = dict(
    max_length=512,
    num_beams=4,
    no_repeat_ngram_size=2,
    early_stopping=True,
    do_sample=False,
    temperature=0.8,
    top_k=10,
    # length_penalty = 0.9,
    forced_bos_token_id=None,
    # eos_token_id= tokenizer.eos_token_id,
    # pad_token_id= tokenizer.pad_token_id
)
PARAMS_FOR_SAMPLING = {'top_k', 'top_p', 'temperature'}


@st.cache()
def cacherando():
    rando = random_number = random.random()
    return rando


def display_app_header(main_txt, sub_txt, is_sidebar=False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:#3c403f  ; padding:15px">
    <h2 style = "color:black; text_align:center;"> {main_txt} </h2>
    <p style = "color:black; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    else:
        st.markdown(html_temp, unsafe_allow_html=True)


def display_side_panel_header(txt):
    """
    function to display minor headers at side panel
    Parameters
    ----------
    txt: str -> the text to be displayed
    """
    st.sidebar.markdown(f'## {txt}')


@st.experimental_singleton(show_spinner=False)  # ttl=1200,
def load_model():
    with initialize(config_path="../../conf", ):
        conf = compose(config_name=CONFIG_NAME,
                       overrides=[f'+checkpoint_path="{CHECKPOINT_PATH}"', f'model.tokenizer_path="{TOKENIZER_PATH}"'])
    tokenizer = AutoTokenizer.from_pretrained(conf.model.tokenizer_path)

    current_conf_path = '/tmp/hparams.yaml'
    with open(current_conf_path, 'w') as writer:
        writer.write(OmegaConf.to_yaml(conf))
    pl_module = PretrainingModule.load_from_checkpoint(conf.checkpoint_path, hparams_file=current_conf_path,
                                                       strict=True)
    GENERATION_PARAMETERS['eos_token_id'] = tokenizer.eos_token_id
    GENERATION_PARAMETERS['pad_token_id'] = tokenizer.bos_token_id
    pl_module.to(DEVICE)
    return pl_module, tokenizer


@st.cache(max_entries=1)
def setart(df, rando):  # session_id):
    randart = random.randint(0, len(df))
    return randart


def get_position_ids(schema):
    num_sentences = len(schema)
    guide_len = np.random.randint(10, 20, 1)[0]
    random_lengths = np.random.randint(guide_len - 2, guide_len + 2, num_sentences)
    position_ids = []

    for l in random_lengths:
        position_ids.extend(reversed(list(range(l))))
    return torch.LongTensor(position_ids)


def get_decoder_rhyming_ids(decoder_position_ids, schema, tokenizer):
    prev_rhyme_token_mask = np.argwhere(decoder_position_ids == 2)[0]  ## mask for tokens preceedings a rhyming token

    rhyming_ids = torch.zeros_like(decoder_position_ids)
    encoded_schema_token = [tokenizer.encode(RHYME_TOKENS[ord(x) - ord('A')], add_special_tokens=False)[0] for x in
                            schema]
    for i, tm in enumerate(prev_rhyme_token_mask):
        rhyming_ids[tm.item()] = encoded_schema_token[i]
    return rhyming_ids


def generate_text(model, prompt, tokenizer):
    batch = tokenizer(prompt, max_length=512, truncation=True, return_special_tokens_mask=True)
    input_ids = torch.LongTensor(batch['input_ids']).unsqueeze(0).to(DEVICE)
    print('=' * 40)
    print('Input ids:', input_ids[0])
    print('Input string:', tokenizer.decode(input_ids[0]))
    # st.write(str(GENERATION_PARAMETERS))
    batch = None
    pprint(GENERATION_PARAMETERS)
    generation = generate(model=model, in_data=input_ids, schema=None, batch=batch, **GENERATION_PARAMETERS)
    generation = tokenizer.decode(generation[0], remove_special_tokens=False)
    generation = generation.replace(SENTENCE_END, '\n')
    generation = generation.replace(BLOCK_END, '\n\n')
    generation = generation.replace('<s>', '').replace('</s>', '')
    print('Output', generation)

    return generation


def main():
    st.set_page_config(page_title='LyricsBot')  # layout='wide', initial_sidebar_state='auto'
    main_txt = """🎸 🥁 Full Lyrics Bot 🎤 🎧"""
    sub_txt = "Model Trained on WASABI Data"
    display_app_header(main_txt, sub_txt, is_sidebar=False)
    # st.markdown(subtitle)
    # session_id = ReportThread.get_report_ctx().session_id
    st.text_input('Song Title (Type in your own!):', key='songtitle')
    st.text_input('Artist Style:', key='artist')
    st.text_input('Genre:', key='genre')
    st.text_input('Year:', key='chronological_tag')
    st.text_input('Emotions (separated by comma if more than one):', key='emotions')
    st.text_input('Topics (separated by comma if more than one):', key='keywords')
    if 'prev_verses' not in st.session_state:
        st.session_state.prev_verses = ''
    # st.text_input('Rhyming Schema (e.g., A B A C):', key='schema')
    prompt = ''
    if st.session_state.songtitle != '':
        prompt += TITLE + st.session_state.songtitle.strip()
    if st.session_state.artist != '':
        prompt += ARTIST + st.session_state.artist.strip()
    if st.session_state.genre != '':
        prompt += GENRE + st.session_state.genre.strip()
    if st.session_state.emotions != '':
        prompt += EMOTIONS + st.session_state.emotions.strip()
    if st.session_state.keywords != '':
        prompt += TOPICS + st.session_state.keywords.strip()
    # schema = None
    # encoded_schema = None
    # if st.session_state.schema != '':
    #     schema = st.session_state.schema.strip()
    #     encoded_schema = [RHYME_TOKENS[ord(x) - ord('A')] for x in schema.split()]
    #     prompt_schema = SCHEMA + ' '.join(encoded_schema)
    st.session_state.prompt = prompt
    display_side_panel_header("Lyrics Bot!")
    st.sidebar.text_input('Device', DEVICE, disabled=True)
    display_side_panel_header("Configuration")
    do_sample = None
    for k, v in GENERATION_PARAMETERS_INFO.items():
        kind = v[-1]
        label = k.replace('_', ' ').capitalize()
        disabled = False
        if do_sample is not None and not do_sample and k in PARAMS_FOR_SAMPLING:
            disabled = True

        if kind == 'slider':
            if label.lower() == 'temperature':
                label = label + ' (the higher the weirder)'
            GENERATION_PARAMETERS[k] = st.sidebar.slider(label, v[0], v[1], v[2], disabled=disabled)
        elif kind == 'selectbox':
            if k == 'do_sample':
                do_sample = st.sidebar.selectbox(label, v[0], v[1], disabled=disabled)
            GENERATION_PARAMETERS[k] = do_sample
    with st.spinner('Loading model...'):
        model, tokenizer = load_model()
        GENERATION_PARAMETERS['decoder_start_token_id'] = tokenizer.bos_token_id
    generated = None

    if st.button('Generate!'):
        with st.spinner(
                "Generating... please be patient, this can take quite a while. If you adjust anything, you may need to start from scratch."):
            start = time.time()
            generated = generate_text(model, st.session_state.prompt, tokenizer)
            end = time.time()
            st.session_state.ttg = str(round(end - start)) + "s"
            st.header("Generated song")
            st.session_state.gentext = generated.replace(st.session_state.prompt, '')
            st.markdown("⏲️ Time To Generate: " + st.session_state.ttg)

            st.markdown(
                st.session_state.gentext[st.session_state.gentext.index('<lyrics>') + len('<lyrics>'):].replace('\n',
                                                                                                                '<br>'),
                unsafe_allow_html=True)
            print('porcodio2',
                  st.session_state.gentext[st.session_state.gentext.index('<lyrics>') + len('<lyrics>'):].replace('\n',
                                                                                                                  '<br>'))


if __name__ == "__main__":
    main()
