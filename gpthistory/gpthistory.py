import click
import json
import os
import pandas as pd
import logging
from gpthistory.helpers import extract_text_parts, generate_embeddings, calculate_top_titles

# Define the path to the index file in the user's home directory
INDEX_PATH = os.path.join(os.path.expanduser('~'), '.gpthistory', 'chatindex.csv')

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def main():
    """
    Simple CLI for searching within a chat data
    """
    pass

@main.command()
@click.option('--file', type=click.Path(exists=True), help='Input file')
def build_index(file):
    """
    Build an index from a given chat data file
    """
    # TODO: Implement index building
    # Write the index to the predefined path
    # Make sure the directory exists
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    # Load the chat data from the given file
    with open(file) as f:
        data = json.load(f)
    
    chat_ids = []
    section_ids = []
    texts = []
    for entry in data:
        for k, v in entry['mapping'].items():
            text_data = extract_text_parts(v)
            if len(text_data) > 0 and text_data[0] != '':
                # Add relevant chat information to the index
                chat_ids.append(entry['id'])
                section_ids.append(k)
                texts.append(text_data[0])
    logger.info(f"Index built and stored at: {INDEX_PATH}")
    logger.info(f"Conversations indexed: {len(chat_ids)}")
    df = pd.DataFrame({'chat_id': chat_ids, 'section_id': section_ids, 'text': texts})
    df = df[~df.text.isna()] 
    df['id'] = df['chat_id']
    df.set_index("id", inplace=True)

    # Handle incremental index updates
    current_df = pd.DataFrame()    
    rows_only_in_df = pd.DataFrame()
    incremental = False
    if os.path.exists(INDEX_PATH):
        incremental = True
        current_df = pd.read_csv(INDEX_PATH, sep='|')
        current_df['id'] = current_df['chat_id']
        current_df.set_index("id", inplace=True)
        # Use merge with indicator=True to find rows present in one DataFrame but not the other
        merged_df = df.merge(current_df, how='outer', indicator=True)
        # Query rows only present in df1
        rows_only_in_df = merged_df.query('_merge == "left_only"').drop(columns='_merge')
    else:
        rows_only_in_df = df
    
    if incremental and len(rows_only_in_df) > 0:
        logger.info("Only generating embeddings for new conversations to save money.")
    
    # Generate and add embeddings to the index
    embeddings = generate_embeddings(rows_only_in_df.text.tolist())
    rows_only_in_df['embeddings'] = embeddings
    final_df = pd.concat([rows_only_in_df, current_df])
    logger.info(f"Total conversations: {len(final_df)}")
    final_df.to_csv(INDEX_PATH, sep='|', index=False)

@main.command()
@click.argument('keyword', required=True)
def search(keyword):
    """
    Search a keyword within the index
    """
    # TODO: Implement search function
    # Load the index from the predefined path
    logger.info("Searching for keyword: %s", keyword)
    if os.path.exists(INDEX_PATH):
        df = pd.read_csv(INDEX_PATH, sep='|')
        df['embeddings'] = df.embeddings.apply(lambda x: [float(t) for t in json.loads(x)])
        filtered = df[df.text.str.contains(keyword)]
        
        # Calculate top titles and their corresponding chat IDs
        chat_ids, top_titles, top_scores = calculate_top_titles(df, keyword)
        
        for i, t in enumerate(top_titles):
            logger.info("%s: %s", chat_ids[i], t)
            logger.info("ChatGPT Conversation link: https://chat.openai.com/c/%s", chat_ids[i])
            logger.info("--------------------------------------")
    else:
        click.echo("Index not found. Please build the index first.")
        return

if __name__ == "__main__":
    main()
