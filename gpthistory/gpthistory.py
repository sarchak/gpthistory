import typer
import json
import os
import pandas as pd
from rich import print
from gpthistory.helpers import (
    extract_text_parts,
    generate_embeddings,
    calculate_top_titles,
)

main = typer.Typer()

# Define the path to the index file in the user's home directory
INDEX_PATH = os.path.join(os.path.expanduser("~"), ".gpthistory", "chatindex.csv")


@main.command()
def build_index(file: typer.FileText):
    """
    Build an index from a given chat data file xxx
    """
    # Make sure the directory exists
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    # Load the chat data from the given file
    data = json.load(file)

    chat_ids = []
    section_ids = []
    texts = []
    for entry in data:
        for k, v in entry["mapping"].items():
            text_data = extract_text_parts(v)
            if len(text_data) > 0 and text_data[0] != "":
                # Add relevant chat information to the index
                chat_ids.append(entry["id"])
                section_ids.append(k)
                texts.append(text_data[0])
    print(f"[cyan]Index built and stored at:[/cyan] {INDEX_PATH}")
    print(f"[cyan]Conversations indexed:[/cyan] {len(chat_ids)}")
    df = pd.DataFrame({"chat_id": chat_ids, "section_id": section_ids, "text": texts})
    df = df[~df.text.isna()]
    df["id"] = df["chat_id"]
    df.set_index("id", inplace=True)

    # Handle incremental index updates
    current_df = pd.DataFrame()
    rows_only_in_df = pd.DataFrame()
    incremental = False
    if os.path.exists(INDEX_PATH):
        incremental = True
        current_df = pd.read_csv(INDEX_PATH, sep="|")
        current_df["id"] = current_df["chat_id"]
        current_df.set_index("id", inplace=True)
        # Use merge with indicator=True to find rows present in one DataFrame but not the other
        merged_df = df.merge(current_df, how="outer", indicator=True)
        # Query rows only present in df1
        rows_only_in_df = merged_df.query('_merge == "left_only"').drop(
            columns="_merge"
        )
    else:
        rows_only_in_df = df

    if incremental and len(rows_only_in_df) > 0:
        print(
            "[yellow]Only generating embeddings for new conversations to save money.[/yellow]"
        )

    import pickle

    with open("convos.pkl", "wb") as f:
        pickle.dump(rows_only_in_df, f)

    # Generate and add embeddings to the index
    embeddings = generate_embeddings(rows_only_in_df.text.tolist())
    rows_only_in_df["embeddings"] = embeddings
    final_df = pd.concat([rows_only_in_df, current_df])
    print(f"[cyan]Total conversations:[/cyan] {len(final_df)}")
    final_df.to_csv(INDEX_PATH, sep="|", index=False)


@main.command()
def search(keyword: str, topk: int = 5, thr: float | None = None):
    """
    Search a keyword within the index with an optional threshold argument.
    """
    print(f"[cyan]Searching for:[/cyan] '{keyword}'")
    if os.path.exists(INDEX_PATH):
        df = pd.read_csv(INDEX_PATH, sep="|")
        df["embeddings"] = df.embeddings.apply(
            lambda x: [float(t) for t in json.loads(x)]
        )
        filtered = df[df.text.str.contains(keyword)]

        if filtered.shape[0] == 0:
            print(
                "[yellow]No exact matches found. Performing solely embedding search.[/yellow]"
            )
            filtered = df.copy()

        # Calculate top titles and their corresponding chat IDs based on the threshold
        chat_ids, top_titles, top_scores = calculate_top_titles(
            filtered, keyword, thr, topk
        )

        for i, t in enumerate(top_titles):
            print(
                f"""\
--------------------------------------------------------------------------------
[cyan bold]url:[/cyan bold] [green]https://chat.openai.com/c/{chat_ids[i]}[/green]
[cyan bold]score:[/cyan bold] {top_scores[i]:.2f}

{t}
-------------------------------------------------------------------------------\
"""
            )
    else:
        print("Index not found. Please build the index first.")
        return


if __name__ == "__main__":
    main()
