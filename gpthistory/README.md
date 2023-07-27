# gpthistory

`gpthistory` is a Python package that provides a powerful tool for indexing and searching ChatGPT conversations. This package allows users to build an index from chat data files, generate embeddings for efficient searching, and perform searches to find relevant conversations based on keywords.

## Installation

You can easily install `gpthistory` via pip:

```bash
pip install gpthistory
```

## Indexing and Search

### Indexing

The `build_index` command allows you to build an index from your chat data files. The tool extracts relevant text parts from each chat entry and stores them in the index along with their associated chat IDs and section IDs.

To build an index, run:

```bash
gpthistory build_index --file /path/to/conversations.json
```

Replace `/path/to/conversations.json` with the path to your chat data file in JSON format.

### Searching

Once you have built the index, you can perform searches using the `search` command. The tool takes a keyword as input and returns the top matching conversations from the index and also the conversation history link so that you can directly go to that link.

To search for a keyword, run:

```bash
gpthistory search "your_keyword"
```

Replace `"your_keyword"` with the keyword you want to search for.

The search algorithm uses embeddings to efficiently match the keyword against the indexed text parts. It calculates dot product scores between the query embedding and all embeddings in the index. Conversations with dot product scores above a certain threshold are considered as top matches.

## Example Usage

```bash
# Build the index from conversations.json
gpthistory build_index --file conversations.json

# Search for conversations related to "chatbot"
gpthistory search "chatbot"
```

## License

`gpthistory` is distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Author

Your Name
Email: shrikar84@gmail.com

## Feedback and Contributions

We welcome feedback and contributions to improve `gpthistory`. If you encounter any issues, have suggestions, or want to contribute, please create an issue or submit a pull request on our [GitHub repository](https://github.com/sarchak/gpthistory).

## Disclaimer

Please note that this tool is intended for research and educational purposes. Make sure you have proper permissions and adhere to the usage terms and conditions of the data sources you analyze with this tool.

