# Importar bibliotecas necessárias
import streamlit as st
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from langchain_community.document_loaders import YoutubeLoader
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from tqdm import tqdm

# Configurar as chaves de API
key = st.secrets["KEYYOUTUBE"]  # Insira sua chave API do YouTube
key_openia = st.secrets["KEYOPENIA"]  # Insira sua chave API do OpenAI

# Funções
def get_video_id(url):
    parsed_url = urlparse(url)
    query_string = parse_qs(parsed_url.query)
    return query_string.get('v', [None])[0]

def get_channel_id_by_video_id(video_id):
    youtube = build('youtube', 'v3', developerKey=key)
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    return response['items'][0]['snippet']['channelId'] if response['items'] else None

def get_latest_videos_links(channel_id, max_results):
    youtube = build('youtube', 'v3', developerKey=key)
    request = youtube.search().list(part='snippet', channelId=channel_id, maxResults=max_results, order='date', type='video')
    response = request.execute()
    return [(item['snippet']['title'], f'https://www.youtube.com/watch?v={item["id"]["videoId"]}') for item in response['items']]

def get_corpus_from_url_list(url_list):
    corpus = ""
    for url in url_list:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language=["pt"])
        loaded_data = loader.load()
        if loaded_data:
            corpus += loaded_data[0].page_content
    return corpus

def divide_string(s, tamanho_max=3000):
    return [s[i:i+tamanho_max] for i in range(0, len(s), tamanho_max)]

# Interface Streamlit
st.title("Resumo de Nicho de Canal do YouTube")

url = st.text_input("Insira a URL do vídeo do YouTube:")
if st.button("Analisar"):
    if url:
        video_id = get_video_id(url)
        channel_id = get_channel_id_by_video_id(video_id)
        latest_videos = get_latest_videos_links(channel_id, 10)

        if latest_videos:
            st.write("Vídeos mais recentes:")
            for title, video_url in latest_videos:
                st.write(f"[{title}]({video_url})")

            url_list = [video[1] for video in latest_videos]
            corpus = get_corpus_from_url_list(url_list)
            document_chunks = divide_string(corpus)

            summaries = []
            llm = OpenAI(temperature=0.6, openai_api_key=key_openia, max_tokens=500)
            for chunk in tqdm(document_chunks[:10], desc="Gerando Resumos"):
                youtube_summary_template = PromptTemplate(
                    input_variables=['transcricao'],
                    template="Com base na seguinte transcrição, ```{transcricao}```, descreva o foco principal do canal de YouTube."
                )
                youtube_summary = LLMChain(llm=llm, prompt=youtube_summary_template)

                niche_summary_template = PromptTemplate(
                    input_variables=['transcricao'],
                    template="Resuma em no máximo 3 palavras o nicho do seguinte canal: ```{youtube_summary}```"
                )
                niche_summary = LLMChain(llm=llm, prompt=niche_summary_template)

                overall_chain = SimpleSequentialChain(chains=[youtube_summary, niche_summary])
                summaries.append(niche_summary.run(chunk))

            niche_summary_list_template = PromptTemplate(
                input_variables=['summary'],
                template="Com base nesses tópicos ```{summary}``` liste de 3 a 5 principais nichos de canal de youtube"
            )

            niche_summary_list = LLMChain(llm=llm, prompt=niche_summary_list_template)
            niches = niche_summary_list.run(" ".join(summaries))

            st.write("Nicho do canal:", niches)
        else:
            st.write("Nenhum vídeo encontrado para o canal.")
    else:
        st.write("Por favor, insira uma URL válida.")
