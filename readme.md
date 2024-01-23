The clustering algorithm goes through the following steps:

1. Clean up the original csv files by making it properly double quoted and removing extra newlines.
2. Extract webpage text by URL inside csv files.
3. Clean webpage text files by removing empty files (unable to connect for security reasons), and remove text grabbed from youtube sites (for they are not informative about the project)
4. Generate raw tags based on LLM (RoBERTa-large)
5. Compute similarity score between each webpage text and tags to form similarity vectors for each files
6. Do PCA projection for files based on their similarity vectors

Python file involved in each step:

1. Processed by hand.
2. extract_webpage.py
3. Processed by hand (sorting by file size is helpful).
4. keyword_extraction.py
5. classification.py
6. graph.py

Configuration:

1. Download RoBERTa-large from https://huggingface.co/roberta-large and place it under modules/roberta-large/
2. To run graph.py, type in command line: streamlit run graph.py