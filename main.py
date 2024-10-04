import os
import streamlit as st
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from langchain_community.document_loaders import YoutubeLoader
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from tqdm import tqdm

# Configurações de chave de API
key_youtube = st.secrets['keyyoutube']
key_openai = st.secrets['keyopenia']

# Função para obter vídeos recentes de um canal
def get_latest_videos_links(channel_id, max_results=10):
    youtube = build('youtube', 'v3', developerKey=key_youtube)
    request = youtube.search().list(
        part='snippet',
        channelId=channel_id,
        maxResults=max_results,
        order='date',
        type='video'
    )
    response = request.execute()

    videos = [(item['snippet']['title'], f'https://www.youtube.com/watch?v={item["id"]["videoId"]}') for item in response['items']]
    return videos

# Função para obter o ID do canal baseado no vídeo
def get_channel_id_by_video_id(video_id):
    youtube = build('youtube', 'v3', developerKey=key_youtube)
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()

    if response['items']:
        return response['items'][0]['snippet']['channelId']
    return None

# Função para extrair o ID do vídeo de uma URL
def get_video_id(url):
    parsed_url = urlparse(url)
    return parse_qs(parsed_url.query).get('v', [None])[0]

# Função para carregar o conteúdo de vídeos de uma lista de URLs
def get_corpus_from_url_list(url_list):
    corpus = ""
    for url in url_list:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["pt"])
        loaded_data = loader.load()
        if loaded_data:
            corpus += loaded_data[0].page_content
        else:
            st.warning(f"Não foi possível carregar conteúdo do vídeo: {url}")
    return corpus

# Função para dividir uma string em pedaços menores
def divide_string(text, chunk_size=3000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Função principal de sumarização e identificação de nicho
def summarize_niche(document_chunks, llm):
    summaries = []

    for chunk in tqdm(document_chunks[:10], desc="Processando"):
        youtube_summary_template = PromptTemplate(
            input_variables=['transcricao'],
            template="Baseado nessa transcrição ```{transcricao}``` defina em apenas uma frase o nicho desse canal de youtube"
        )
        youtube_summary = LLMChain(llm=llm, prompt=youtube_summary_template)

        Niche_summary_template = PromptTemplate(
            input_variables=['transcricao'],
            template="Resuma em no máximo 3 palavras o nicho do seguinte canal: ```{youtube_summary}```"
        )
        niche_summary = LLMChain(llm=llm, prompt=Niche_summary_template)

        overall_chain = SimpleSequentialChain(chains=[youtube_summary, niche_summary])
        summaries.append(niche_summary.run(chunk))

    return summaries

# Configurações da página no Streamlit
st.title("Extrator de Nicho de Canais do YouTube")
st.write("Este aplicativo extrai e sumariza o nicho de um canal com base nos vídeos mais recentes.")

# Entrada da URL do vídeo do YouTube
youtube_url = st.text_input("Insira a URL de um vídeo do YouTube:")

if youtube_url:
    video_id = get_video_id(youtube_url)
    
    if video_id:
        channel_id = get_channel_id_by_video_id(video_id)

        if channel_id:
            # Obter vídeos recentes do canal
            st.write(f"ID do Canal: {channel_id}")
            latest_videos = get_latest_videos_links(channel_id)
            st.write("Últimos vídeos encontrados:")
            st.write(latest_videos)

            # Obter conteúdo de vídeos
            url_list = [url for _, url in latest_videos]
            corpus = get_corpus_from_url_list(url_list)

            # Dividir o conteúdo e processar a sumarização
            document_chunks = divide_string(corpus)
            llm = OpenAI(temperature=0.6, openai_api_key=key_openai, max_tokens=500)
            summaries = summarize_niche(document_chunks, llm)

            # Mostrar o resultado
            st.write("Sumários de nichos:")
            st.write(summaries)
        else:
            st.error("Não foi possível encontrar o ID do canal.")
    else:
        st.error("URL inválida.")
