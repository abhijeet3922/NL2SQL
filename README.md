# Chat with Tables: Query tabular data in English using self-hosted Large Language Models

Business users and non-technical professionals often need to quickly analyse or transform tabular data in spreadsheets for ad hoc business intelligence. However, they might lack the necessary programming knowledge to do so themselves and therefore must reach out to a data analyst. Such unexpected delays have the potential to incur huge opportunity costs for time-sensitive business decisions which must be informed by accurate analysis of data.

Generative AI powered by Large Language Models (LLMs) is being used to create novel text, images, and even videos. LLMs specialising in generating code are already being used in enterprise solutions like GitHub Copilot, Gemini Code Assist by Google, watsonx by IBM, and Amazon Q Developer (previously Amazon CodeWhisperer) to boost productivity for developers and programmers. Along the same lines, there now exist LLMs specialising in generating Structured Query Language (SQL), which is widely used across enterprise domains to manage databases and analyse and transform tabular data.

## Workshop Objective
In this workshop, we will scratch the idea of fetching and analyzing data using natural language. We demonstrate how to accurately do a quick proof of concept by creating an streamlit application using Ollama endpoints to analyse and query CSV files. We also discuss challenges and techniques to overcome these challenges to a certain extent.

## Outline
1. Quick overview of the workshop
2. Chapter-1: Converting natural language to SQL using Code LLM for SQL Table.
3. Discussion on running Quantized LLMs locally for memory constraints and data privacy.
4. Chapter-2: Hands-on: Setting up Ollama model server
5. Chapter-3: Metadata pruning for Large Table. Helpful for reducing hallucination/Confusion by reducing prompt length.
6. Chapter-4: Hands-on: Data processing techniques for correcting LLM Hallucinations using Static Analysis with sqlglot.
7. Hands-on: Setting up Streamlit and building quick interactive front-end applications
8. Discussion on how to create generic "Chat with X" capabilities

### Intended Audience
This workshop is intended for *data enginners, data scientists, and researchers* with *basic Python experience* who are working on Generative AI use-cases and want to leverage enterprise data. This might also interest *business analysts or business consumers* who require data querying and analysis services regularly.

Overall, any professional with at least some experience with Python programming who is interested in getting started with Gen AI will stand to benefit from this workshop since it covers both the end-to-end data pipeline as well how to prepare a demo-worthy front-end user interface.

## Takeaways
1. How to analyse tabular data in CSV format using English language queries.
2. How to run LLMs locally or within your organisation network using Ollama.
3. How to quickly develop interactive web applications using Streamlit.
4. How to create “Chat with X” applications for other data formats.

## Pre-Reading for Workshop
Here are some links if you are interested in having pre-read about libraries and models being utilized for the workshop.  

1. [NL2SQL Model](https://huggingface.co/defog/sqlcoder-7b-2): Open source model for NL2SQL. One can use GPT4, Copilot, Amazon Q etc.
2. [OLLAMA](https://github.com/ollama/ollama): Setting up language model endpoints with memory constraints.
3. [Embedding Model - MixedBread](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1): The crispy sentence embedding family from mixedbread ai. We will use embeddings for metadata pruning.
4. [SQLGLOT](https://sqlglot.com/sqlglot.html): SQLGlot is a no-dependency SQL parser, transpiler, optimizer, and engine. It aims to read a wide variety of SQL inputs (21 different dialects) and output syntactically and semantically correct SQL in the targeted dialects.