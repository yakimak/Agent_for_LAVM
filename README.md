---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


---



What a challenge. I final got the certificate of the Hugging Face Agents Course!

techstack: supabase, tavily, websearch, langgraph, Qwen, hf embeddings, Multi-step reasoning with tool coordination 
(interpret images, transcribe audios, do web searchs, do wikipedia searchs, execute Python code and let the agent know about the result, read excel files and even ask the agent questions about Youtube videos)

This wasn't the most difficult thing though, the hardest part came when I had to overcome the GAIA benchmarks. For those who don't know, this benchmark consists of the agent having to answer exactly what is asked, whether it is a number, a number; whether it is a list of items, a list of items; whether it is a word, a word, etc. Without all the pantomime that the agent usually adds. With this course, I realized that the agents' capabilities are greater than I thought.

Huge thanks to the HF team for an incredibly hands on program!
Many thanks to all great people behind this course, e. g. Thomas Simonini, Sergio Paniego Blanco, Joffrey THOMAS, Ben Burtenshaw and many more! Thanks also to the amazing community which is always helpful, e. g. on Discord, to solve problems.

building real world AI agents
sharpening prompt engineering & tool calling skills, or🔹 simply seeing how far you can push today’s open source LLM stackto dive in, experiment, and share your learnings. 🛠️

🎯 GAIA Benchmark Mastery: GAIA (General AI Assistants) is Hugging Face's rigorous evaluation framework that tests agents on real-world tasks requiring complex reasoning, multi-modal understanding, and precise tool usage. While humans achieve ~92% accuracy and GPT-4 with plugins only reaches ~15%, I'm proud to have exceeded the minimum requirements and am planning further optimizations to push the boundaries! 💪

So still work to do to further improve my code.

#AI #AIAgents #HuggingFace #GAIA #MachineLearning #LangGraph #AgenticWorkflows #OpenSource




---
Building an AI Agent on HF 🤗 

I've been working on creating AI agents that meet the challenging GAIA benchmark standards. 
GAIA tests agents on real-world tasks that need complex reasoning, image understanding, and precise tool use. While GPT-4 with plugins only reaches ~15% accuracy on these tests, I'm proud to have exceeded the minimum requirements.

My main goal wasn't about earning a certificate, but understanding real-world agent requirements and developing my skills through open source contributions and challenging projects.

 🛠️ Tech stack:
- Vector DB: Supabase with pgvector
- Search: Tavily API, Wikipedia, Arxiv
- LLM: Qwen (+ groq, openai, deepseek)
- Embeddings: HuggingFace all-mpnet-base-v2
- Agent Framework: LangGraph, LangChain, LangSmith (monitoring)
- Techniques: Chain of Thought, Tool Coordination
- Multi-modal processing (image/audio/documents)

📚 Custom Tools Developed:
- SearchTools: Web search, Wiki search, ArXiv search, similarity search
- MathTools: Basic arithmetic, modulus, power, square root operations
- CodeTools: Multi-language executor (Python, Bash, SQL, C, Java)
- DocsTools: File reader, downloader, dataframe analyzer, OCR extractor
- ImageTools: Image analysis, transformation, drawing, generation

🎯 Key capabilities developed:
- Multi-step reasoning & complex query understanding
- Web browsing & real-time information retrieval
- Image analysis & OCR extraction
- Audio transcription & processing & YouTube video analysis
- Multi-modal file handling (PDF, Excel, CSV)
- Code execution (Python, Bash, SQL, C, Java)
- Database queries & vector similarity search
- Mathematical calculations
- Wikipedia & ArXiv paper search
- Document summarization & data extraction
- Chain of Thought reasoning implementation
- Local testing & evaluation scripts for GAIA benchmarks

The hardest part? Meeting GAIA's strict requirements; giving EXACTLY what's asked (just a number, word, or list) without the usual AI explanations.
This project maed me realized that the agents' capabilities are greater than I thought.


Huge thanks to the HF team for an incredibly hands on program!
Many thanks to all great people behind this course; Thomas Simonini, Sergio Paniego Blanco, Joffrey THOMAS, Ben Burtenshaw and many more! Thanks also to the amazing community which is always helpful, on Discord, to solve problems.

Now working on further improvements to push the boundaries🤓