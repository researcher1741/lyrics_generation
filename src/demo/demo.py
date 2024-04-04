import streamlit as st
import random
import time
import torch
from src.lyrics_generation.generate import generate

from src.lyrics_generation.encoder_decoder_module import EncoderDecoderModule
from src.lyrics_generation_utils.utils import ARTIST, EMOTIONS, GENRE, LYRICS, TITLE, TOPICS

DEVICE = 'cuda:0'
# DEVICE='cpu'
CHECKPOINT_PATH = 'experiments/genius_full_song/gpt2-medium.train-wa.finetune-g/checkpoints/epoch=9-step=25800.ckpt'
MODEL_NAME = 'gpt2-medium'
GENERATION_PARAMETERS_INFO = dict(
    max_length=(100, 1024, 512, 'slider'),
    num_beans=(1, 5, 2, 'slider'),
    no_repeat_ngram_size=(1, 10, 2, 'slider'),
    early_stopping=([True, False], 0, 'selectbox'),
    temperature=(1.0, 5.0, 1.0, 'slider'),
    length_penalty=(0.0, 1.0, 1.0, 'slider'),
    top_k=(0, 100, 50, 'slider'),
    top_p=(0.0, 1.0, 1.0, 'slider')

)
GENERATION_PARAMETERS = dict(
    max_length=1024,
    num_beams=2,
    no_repeat_ngram_size=2,
    early_stopping=True,
    do_sample=True,
    temperature=0.8,
    top_k=10,
    length_penalty=0.9

)


@st.cache()
def cacherando():
    rando = random_number = random.random()
    return rando


def display_app_header(main_txt, sub_txt, is_sidebar=False):
    """
    function to display major headers at user interface
    Parameters
    Inputs:
        - main_txt: str -> the major text to be displayed
        - sub_txt: str -> the minor text to be displayed
        - is_sidebar: bool -> check if its side panel or major panel
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


@st.cache(allow_output_mutation=True, max_entries=1, show_spinner=False)  # ttl=1200,
def load_model():
    from omegaconf import OmegaConf
    current_conf_path = '/tmp/hparams.yaml'
    conf = torch.load(CHECKPOINT_PATH, map_location='cpu')['hyper_parameters']
    TOKENIZER_PATH = '/'.join(CHECKPOINT_PATH.split('/')[:-2]) + '/tokenizer/'
    conf['model']['tokenizer_path'] = TOKENIZER_PATH
    with open(current_conf_path, 'w') as writer:
        writer.write(OmegaConf.to_yaml(conf))
    pl_module = EncoderDecoderModule.load_from_checkpoint(CHECKPOINT_PATH, hparams_file=current_conf_path)
    pl_module.eval()
    pl_module.to(DEVICE)
    return pl_module, tokenizer


@st.cache(max_entries=1)
def setart(df, rando):  # session_id):
    randart = random.randint(0, len(df))
    return randart


@st.cache(max_entries=1)
def settitle():  # (session_id):
    sampletitles = [
        "Love Is A Vampire",
        "The Cards Are Against Humanity",
        "My Grandmother Likes My Music",
        "Call A Doctor, It Is Urgent",
        "So, That Just Happened",
        "Dogs Versus Cats",
        "Parties Are Overrated",
        "I Believe That Is Butter",
        "Panic In The Grocery Store",
        "He's Not A Suspect Yet",
        "Jumping To The Moon",
        "My Father Enjoys Scotch",
        "My Goodness You Are Silly",
        "The Clouds Are Bright",
        "I Love People",
        "Notorious Kangaroos"
    ]
    randtitle = random.choice(sampletitles)
    return randtitle


def generate_text(model, prompt, tokenizer):
    batch = tokenizer(prompt, max_length=1024, truncation=True, return_special_tokens_mask=True)
    input_ids = torch.LongTensor(batch['input_ids']).unsqueeze(0).to(DEVICE)
    # st.write(str(GENERATION_PARAMETERS))
    generation = generate(model, input_ids, **GENERATION_PARAMETERS)
    generation = tokenizer.decode(generation[0])
    return generation


def main():
    st.set_page_config(page_title='LyricsBot')  # layout='wide', initial_sidebar_state='auto'
    main_txt = """🎸 🥁 Lyrics Bot 🎤 🎧"""
    sub_txt = "Model Trained on WASABI Data"
    display_app_header(main_txt, sub_txt, is_sidebar=False)
    # st.markdown(subtitle)
    # session_id = ReportThread.get_report_ctx().session_id
    st.text_input('Song Title (Type in your own!):', key='songtitle')
    st.text_input("Artist: ", key='artist')
    st.text_input('Genre:', key='genre')
    st.text_input('Year:', key='chronological_tag')
    st.text_input('Emotions (separated by comma if more than one):', key='emotions')
    st.text_input('Key words (separated by comma if more than one):', key='keywords')
    prompt = ''

    if st.session_state.songtitle != '':
        prompt += TITLE + st.session_state.songtitle
    if st.session_state.artist != '':
        prompt += ARTIST + st.session_state.artist
    if st.session_state.genre != '':
        prompt += GENRE + st.session_state.genre
    if st.session_state.emotions != '':
        prompt += EMOTIONS + st.session_state.emotions
    if st.session_state.keywords != '':
        prompt += TOPICS + st.session_state.keywords
    prompt += LYRICS
    st.session_state.prompt = prompt
    display_side_panel_header("Lyrics Bot!")
    st.sidebar.text_input('Device', DEVICE, disabled=True)
    display_side_panel_header("Configuration")
    for k, v in GENERATION_PARAMETERS_INFO.items():
        kind = v[-1]
        label = k.replace('_', ' ').capitalize()
        if kind == 'slider':
            if label.lower() == 'temperature':
                label = label + ' (the higher the weirder)'
            GENERATION_PARAMETERS[k] = st.sidebar.slider(label, v[0], v[1], v[2])
        elif kind == 'selectbox':
            GENERATION_PARAMETERS[k] = st.sidebar.selectbox(label, v[0], v[1])

    session_state = st.session_state
    with st.spinner('Loading model...'):
        model, tokenizer = load_model()

    if st.button('Generate My Songs!'):
        with st.spinner(
                "Generating songs, please be patient, this can take quite a while. "
                "If you adjust anything, you may need to start from scratch."):
            start = time.time()
            generated = generate_text(model, session_state.prompt, tokenizer).replace('\n', '<br>')
            end = time.time()
            session_state.ttg = str(round(end - start)) + "s"
        st.header("Generated song")
        # sep = '<|endoftext|>'
        session_state.gentext = generated.replace(session_state.prompt, '')
        st.markdown("⏲️ Time To Generate: " + session_state.ttg)
        st.markdown(session_state.gentext, unsafe_allow_html=True)
    else:
        pass


if __name__ == "__main__":
    main()
