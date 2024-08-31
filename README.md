# YouTube Niche Extractor

This project automates the process of extracting and summarizing the niche of YouTube channels by analyzing their latest video content. Utilizing the YouTube API, Langchain, OpenAI, and TQDM, this tool offers a streamlined way to identify the primary focus areas of any YouTube channel.

## Project Overview

The **YouTube Niche Extractor** is designed to:
1. Extract the latest video links from a specified YouTube channel.
2. Transcribe the video content using Langchain.
3. Summarize the content to identify the channel's niche using OpenAI's language models.

## Key Features
- **Automated Video Extraction:** Retrieves up to 10 of the latest videos from a specified channel.
- **Content Transcription:** Converts video content into text for analysis.
- **Niche Identification:** Uses AI-driven language models to summarize and define the channel’s niche.

## How It Works

1. **Extract Video IDs:** Given a YouTube channel, the script fetches the latest video URLs and IDs.
2. **Transcription:** The video content is loaded and transcribed using the `YoutubeLoader`.
3. **Summarization:** The transcribed content is broken into manageable chunks and fed into OpenAI models to extract niche-related keywords and phrases.
4. **Final Niche Summary:** A final summary of the channel's niche is generated based on the extracted information.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following:
- A Google Colab environment.
- API keys for both YouTube and OpenAI.

### Setup

1. **Clone the Repository**: Import the project into your Google Colab environment.
2. **Install Dependencies**:
   Ensure you have the necessary packages installed:
   ```python
   !pip install google-api-python-client langchain tqdm openai
