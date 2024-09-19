# Complete Guide to Building, Deploying, and Optimizing Generative AI with Langchain and Huggingface

Welcome to the **Complete Guide to Building, Deploying, and Optimizing Generative AI** using **Langchain**, **Huggingface**, and **Streamlit**! This repository will guide you through building and deploying a Generative AI application using these frameworks.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Building a Generative AI with Langchain](#building-a-generative-ai-with-langchain)
- [Integrating Huggingface Transformers](#integrating-huggingface-transformers)
- [Deploying with Streamlit](#deploying-with-streamlit)
- [Optimizing Model Performance](#optimizing-model-performance)
- [Running the Application](#running-the-application)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Generative AI is transforming industries with its ability to generate text, images, and other forms of media. In this guide, we'll use:
- **Langchain**: For managing prompts and creating application chains.
- **Huggingface**: For integrating state-of-the-art models like GPT, BERT, and others.
- **Streamlit**: For building interactive user interfaces and deploying AI applications easily.

## Project Overview

We'll build, deploy, and optimize a Generative AI application:
1. Using **Langchain** for prompt management.
2. Leveraging **Huggingface Transformers** for text generation.
3. Deploying the model using **Streamlit**.
4. Optimizing the model for performance and scalability.

## Prerequisites

Ensure you have the following installed:
- Python 3.9+
- [Langchain](https://github.com/hwchase17/langchain)
- [Huggingface Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- Git
- Docker (optional for deployment)

Basic knowledge of:
- NLP and Generative AI models
- Web deployment and APIs
- Streamlit for creating web apps

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SuchindraKumar/Complete_Generative_AI_With_Langchain_and_Huggingface.git
    cd Complete_Generative_AI_With_Langchain_and_Huggingface
    ```

2. **Create a virtual environment**:
    ```bash
    conda create -p venv python==3.9 -y
    source activate ./venv`
    ```

3. **Install required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install Streamlit**:
    ```bash
    pip install streamlit
    ```

## Building a Generative AI with Langchain

Langchain makes it easy to build powerful applications that use language models. You'll use Langchain to:
- Define templates for generating text.
- Build chains that manage prompts and outputs.

### Example Code
```python
from langchain import PromptTemplate, LLMChain
from transformers import pipeline

template = """Translate this English text to French: {text}"""
prompt = PromptTemplate(template=template, input_variables=["text"])

translator = pipeline('translation_en_to_fr')
llm_chain = LLMChain(prompt_template=prompt, llm=translator)

response = llm_chain.run("Hello, how are you?")
print(response)

