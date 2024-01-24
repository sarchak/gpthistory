# gpthistory

`gpthistory` is a Python package that provides a powerful tool for indexing and searching ChatGPT conversations. This package allows users to build an index from chat data files, generate embeddings for efficient searching, and perform searches to find relevant conversations based on keywords.


## Installation

You can easily install `gpthistory` via pip:

```bash
pip install gpthistory
```

```bash
# Installing from souce
git clone git@github.com:sarchak/gpthistory.git
cd gpthistory
pip install -e .
```

## Download the conversation history
Unfortunately, there is no way to programmatically get the conversation history. As as work around export
the conversations by going to the Setting Section. One you get the email from OpenAI download and unzip the folder
which contains the conversations.json file

<img width="681" alt="SCR-20230726-ugkl" src="https://github.com/sarchak/gpthistory/assets/839293/212e5733-e0cf-4e4b-b45c-e1daaaeeaa4f">

## Setting up OpenAI Key
We use openai embeddings to find semantic similarity. Hence before building index. Make sure you set the OpenAI Key on the shell.

export OPENAI_API_KEY='your open ai key'

## Indexing and Search
[![asciicast](https://asciinema.org/a/ht0KVofl1GZwLgP1SEHKwKzX8.svg)](https://asciinema.org/a/ht0KVofl1GZwLgP1SEHKwKzX8)
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
gpthistory build-index --file conversations.json

# Search for conversations related to "chatbot"
gpthistory search "chatbot"
```

## License

`gpthistory` is distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Author

Your Name
Twitter: [shrikar84](https://x.com/shrikar84)

## Feedback and Contributions

We welcome feedback and contributions to improve `gpthistory`. If you encounter any issues, have suggestions, or want to contribute, please create an issue or submit a pull request on our [GitHub repository](https://github.com/sarchak/gpthistory).

## Disclaimer

Please note that this tool is intended for research and educational purposes. Make sure you have proper permissions and adhere to the usage terms and conditions of the data sources you analyze with this tool.

