from dotenv import load_dotenv
load_dotenv()

import os
from tavily import TavilyClient
from pytubefix import YouTube
from pytubefix.cli import on_progress
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.tools import tool
from langchain_community.document_loaders import ArxivLoader
import whisper
import contextlib
import io
import pandas as pd


@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for the given query and returns 1 result in the format:
    Title: {title}, Source: {source}
    {content}
    Args:
        query (str): The search query.
    """
    docs = WikipediaLoader(query=query, load_max_docs=1).load()[0]
    return f"Title: {docs.metadata['title']}, Source: {docs.metadata['source']}\n{docs.page_content}"

@tool
def web_search(query: str) -> str:
    """
    Search the web real-time for the given query, explore multiple sources, extract relevant content and return maximum 2 results.
    The results are formatted as:
    Title: {title}, Source: {source}
    {content}
    Args:
        query (str): The search query.
    """
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    docs = client.search(query=query, max_results=2, search_depth="advanced")['results']
    formatted_docs = "\n\n---\n\n".join(
        [
            f"Title: {doc['title']}, Source: {doc['url']}\n{doc['content']}"
            for doc in docs
        ]
    )
    return formatted_docs

@tool
def arxiv_search(query: str, summary: bool) -> str:
    """
    Search Arxiv for the given query and return maximum 3 results. If summary is True, include the summary of each document else provide the whole document.
    The results are formatted as:
    Title: {title}, Published: {published}
    {summary/document}
    Args:
        query (str): The search query.
        summary (bool): Whether to include the summary of each document or the whole document.
    """
    docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_docs = "\n\n---\n\n".join(
        [
            f"""Title: {doc.metadata['Title']}, Published: {doc.metadata['Published']}\n
                {"Summary:" if summary else "Content:"}\n
                {doc.metadata['Summary'] if summary else doc.page_content}"""
            for doc in docs
        ]
    )
    return formatted_docs

@tool
def transcribe_audio(file_name: str) -> str:
    """
    Transcribe the audio file using Whisper and return the transcribed text.
    Args:
        file_name (str): Path to the audio file.
    """
    import torch
    model = whisper.load_model("base").to("cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(os.path.join(os.getcwd(), "files", file_name))
    return result['text']

@tool
def read_file(file_name: str) -> str:
    """
    Read the content of a file and return it.
    Args:
        file_name (str): Path to the file.
    """
    try:
        with open(os.path.join(os.getcwd(), "files", file_name), "r") as file:
            return file.read()
    except Exception as e:
        return f"An error occurred while reading the file: {e}"
    
@tool
def transcribe_youtube_audio(youtube_video_url: str) -> str:
    """
    Extract the audio from a YouTube video and return the transcribed text.
    Args:
        youtube_full_url (str): Full URL of the YouTube video.
    """
    try:
        yt = YouTube(youtube_video_url, on_progress_callback=on_progress)
        audio = yt.streams.filter(only_audio=True).first()
        audio.download(filename="youtube.mp3", output_path="files/")
        transcription = transcribe_audio.invoke("youtube.mp3")
        os.remove(os.path.join("files", "youtube.mp3"))
        return transcription
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def inspect_spreadsheet(file_name: str) -> str:
    """
    Returns column names and the first few rows of an Excel or CSV file to inspect and understand its structure.
    Args:
        file_name (str): Name of the Excel or CSV file.
    """
    try:
        file_path = os.path.join(os.getcwd(), "files", file_name)
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file format. Please provide a CSV or Excel file."
        
        return f"Columns: {list(df.columns)}\nSample:\n{df.head(5).to_markdown()}\n\nNote: The code should not contain any import statements, as the necessary libraries are already imported in the execution environment. The code should also not contain file reading or writing operations (e.g excel, csv), as the spreadsheets are already read and available in the execution environment with the variable name 'df' but the file name should be provided in the code_interpreter tool. Use print statement to output the results and make sure to return the final answer in the required format as specified in the final_answer_guidelines tool."
    except Exception as e:
        return f"An error occurred while inspecting the spreadsheet: {e}"

@tool
def code_interpreter(code: str, file_name: str | None) -> str:
    """
    Execute the provided Python code in a secure environment and return the output.
    If the code requires access to an Excel or CSV file for analysis, the file_name parameter should be provided.
    The code can use libraries like pandas (as pd), numpy (as np), math, random, datetime, itertools, collections, and re.
    Args:
        code (str): The Python code to execute.
        file_name (str | None): The name of the Excel or CSV file to be used in the code, if applicable.
    """
    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec_globals = {
                "__builtins__": __builtins__,
                "math": __import__('math'),
                "np": __import__('numpy'),
                "pd": __import__('pandas'),
                "random": __import__('random'),
                "datetime": __import__('datetime'),
                "re": __import__('re'),
                "itertools": __import__('itertools'),
                "collections": __import__('collections'),
            }
            exec_locals = {}
            if file_name:
                file_path = os.path.join(os.getcwd(), "files", file_name)
                if file_name.endswith('.csv'):
                    exec_locals['df'] = pd.read_csv(file_path)
                elif file_name.endswith('.xlsx'):
                    exec_locals['df'] = pd.read_excel(file_path)
                else:
                    return "Unsupported file format. Please provide a CSV or Excel file."
            exec(code, exec_globals, exec_locals)
        result = output.getvalue().strip()
        return result if result else "Code executed successfully with no output."
    except Exception as e:
        return f"Error during code execution: {e}"
