import os
import tiktoken
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="http://oai.hconeai.com/v1",
    default_headers={
        "Helicone-Auth": f"Bearer {os.environ.get('HELICONE_API_KEY')}",
        "Helicone-Property-project": "gpthistory",
    },
)

# Load model
tokenizer = tiktoken.get_encoding("cl100k_base")
EMBEDDING_MODEL = "text-embedding-3-small"

# Define the path to the index file in the user's home directory
INDEX_PATH = os.path.join(os.path.expanduser("~"), ".chatsearch", "chatindex.csv")


def count_tokens(text):
    return len(tokenizer.encode(text))


def get_first_n_tokens(text: str, n: int) -> str:
    tokens = tokenizer.encode(text)
    first_n_tokens = tokens[:n]
    return tokenizer.decode(first_n_tokens)


def extract_text_parts(data):
    """
    Extract text parts from chat data.
    """
    text_parts = []
    message = data.get("message")
    if message:
        content = message.get("content")
        if content and content.get("content_type") == "text":
            text_parts.extend(content.get("parts", []))
    return text_parts


def split_into_batches(array, batch_size):
    """
    Split an array into batches.
    """
    for i in range(0, len(array), batch_size):
        yield array[i : i + batch_size]


def generate_query_embedding(query):
    """
    Generate an embedding for a query using OpenAI API.
    """
    response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def generate_embeddings(conversations):
    """
    Generate embeddings for conversations using OpenAI API.
    """
    embeddings = []
    for i, batch in enumerate(split_into_batches(conversations, 100)):
        # Suppressing logging of individual batch processing for OpenAI requests
        for i, text in enumerate(batch):
            if count_tokens(text) > 8000:
                batch[i] = get_first_n_tokens(text, 8000)
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        tmp_embedding = [r.embedding for r in response.data]
        embeddings += tmp_embedding
    if len(embeddings) > 0:
        print(f"[cyan]Conversations (Chunks):[/cyan] {len(conversations)}")
        print(f"[cyan]Embeddings:[/cyan] {len(embeddings)}")
    else:
        print("[yellow]No new conversations detected[/yellow]")
    return embeddings


def calculate_top_titles(df, query, thr=0.8, top_n=1000):
    """
    Calculate top titles for a given query using embeddings.
    """

    # Extract the embeddings from the DataFrame
    embedding_array = np.array(df["embeddings"].tolist())
    query_embedding = generate_query_embedding(query)
    # Calculate the dot product between the query embedding and all embeddings in the DataFrame
    dot_scores = np.dot(embedding_array, query_embedding)

    # Filter out titles with dot scores below the threshold
    if thr is not None:
        mask = dot_scores >= thr
    else:
        mask = np.ones_like(dot_scores, dtype=bool)

    filtered_dot_scores = dot_scores[mask]
    filtered_titles = df.loc[mask, "text"].tolist()
    filtered_chat_ids = df.loc[mask, "chat_id"].tolist()

    # Sort the filtered titles based on the dot scores (in descending order)
    sorted_indices = np.argsort(filtered_dot_scores)[::-1][:top_n]

    # Get the top N titles and their corresponding dot scores
    chat_ids = [filtered_chat_ids[i] for i in sorted_indices]
    top_n_titles = [filtered_titles[i] for i in sorted_indices]
    top_n_dot_scores = filtered_dot_scores[sorted_indices]

    return chat_ids, top_n_titles, top_n_dot_scores
