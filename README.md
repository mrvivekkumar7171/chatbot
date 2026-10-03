# ChatBot

1. Create a conda environment
```shell
conda create --name chatbot python=3.14.8
conda activate chatbot
```

2. Install dependencies
```shell
pip install -r requirements.txt
```

3. To run docker and check the status
```shell
docker-compose up -d
docker ps
```

4. Run the frontend using streamlit
```shell
streamlit run .\frontend.py
```

5. Visit LangSmith to track and visualize your LangChain applications:
https://smith.langchain.com/

### Features to Add in the Future
1. Use below platform's api key for Low-frequency background tasks such as summarization, labeling/classification chat, metadata generation, extracting entities, rewriting, tagging, etc.
- Google Gemini API
- Groq
2. Use nosql database online using free tier for storing and retrieving data efficiently, especially for unstructured or semi-structured data.
3. Store Word embeddings in a vector database on cloud using free tier for efficient similarity search and retrieval.
4. Generate embedding once and use multiple times for different tasks to save computation and improve performance.
5. Image and audio models
check if there is max terns or not to prevent infinte loop if the model is not able to generate a response. If the model reaches the maximum number of turns, it should stop generating responses and return an appropriate message to the user.
6. find do we can add bash tool in python code to run bash commands and get the output. This can be useful for automating tasks, running scripts, and interacting with the system.