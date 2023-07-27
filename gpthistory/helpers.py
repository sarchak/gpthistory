import json
import os
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Define the path to the index file in the user's home directory
INDEX_PATH = os.path.join(os.path.expanduser('~'), '.chatsearch', 'chatindex.csv')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_parts(data):
    """
    Extract text parts from chat data.
    """
    text_parts = []
    message = data.get('message')
    if message:
        content = message.get('content')
        if content and content.get('content_type') == 'text':
            text_parts.extend(content.get('parts', []))
    return text_parts

def split_into_batches(array, batch_size):
    """
    Split an array into batches.
    """
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]

def generate_query_embedding(query):
    """
    Generate an embedding for a query using OpenAI API.
    """
    response = openai.Embedding.create(
        input=[query],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def generate_embeddings(conversations):
    """
    Generate embeddings for conversations using OpenAI API.
    """
    embeddings = []
    for i, batch in enumerate(split_into_batches(conversations, 100)):
        logger.info(f"Generating Embeddings for batch: {i + 1}")
        response = openai.Embedding.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        tmp_embedding = [row['embedding'] for row in response['data']]
        embeddings += tmp_embedding
    if len(embeddings) > 0:
        logger.info("Conversations (Chunks) = %d", len(conversations))
        logger.info("Embeddings = %d", len(embeddings))
    else:
        logger.info("No new conversations detected")
    return embeddings

def calculate_top_titles(df, query, top_n=1000):
    """
    Calculate top titles for a given query using embeddings.
    """
    # Extract the embeddings from the DataFrame
    embedding_array = np.array(df['embeddings'].tolist())
    query_embedding = generate_query_embedding(query)
    # Calculate the dot product between the query embedding and all embeddings in the DataFrame
    dot_scores = np.dot(embedding_array, query_embedding)

    # Filter out titles with dot scores below the threshold
    mask = dot_scores >= 0.8
    filtered_dot_scores = dot_scores[mask]
    filtered_titles = df.loc[mask, 'text'].tolist()
    filtered_chat_ids = df.loc[mask, 'chat_id'].tolist()

    # Sort the filtered titles based on the dot scores (in descending order)
    sorted_indices = np.argsort(filtered_dot_scores)[::-1][:top_n]

    # Get the top N titles and their corresponding dot scores
    chat_ids = [filtered_chat_ids[i] for i in sorted_indices]
    top_n_titles = [filtered_titles[i] for i in sorted_indices]
    top_n_dot_scores = filtered_dot_scores[sorted_indices]

    return chat_ids, top_n_titles, top_n_dot_scores
