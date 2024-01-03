# LangChain Server and Web Research Assistant

LangChain is a powerful natural language processing library that combines web searching, document summarization, and report generation. The LangChain Server and Web Research Assistant projects utilize this library to provide an API server with a web-based interface, allowing users to perform advanced research tasks seamlessly.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

LangChain Server and Web Research Assistant leverage the LangChain library and FastAPI framework to deliver a comprehensive solution for users looking to conduct research, summarize documents, and generate detailed reports. The server exposes endpoints for various research tasks, while the web-based interface provides an intuitive way for users to interact with the system.

## Features

- Web searches using DuckDuckGo Search API
- Document summarization using LangChain's ChatOpenAI model
- Report generation based on research summaries
- FastAPI for handling HTTP requests
- Jinja2Templates for rendering HTML responses

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd langchain-server-web-research-assistant
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment by creating a `.env` file with your OpenAI API key:

    ```env
    OPENAI_API_KEY=your-openai-api-key
    ```

4. Run the server:

    ```bash
    uvicorn main:app --reload
    ```

The server will be running at `http://localhost:8000`.

## Usage

LangChain Server and Web Research Assistant provide a web-based interface for users to interact with its functionalities. Users can perform web-based research tasks, summarize documents, and generate detailed research reports.

## Endpoints

### 1. Home

- **URL**: `/`
- **Method**: `GET`
- **Description**: Home endpoint serving the main HTML interface.

### 2. ArXiv Research Assistant

- **URL**: `/arxiv-research-assistant`
- **Method**: `POST`
- **Description**: Endpoint for performing web-based research tasks.
- **Request Parameters**:
    - `question` (form parameter): The user's research question.
- **Response**: HTML rendering of the research results.

### 3. Web Research Assistant

- **URL**: `/web-research-assistant`
- **Method**: `POST`
- **Description**: Endpoint for performing web-based research tasks.
- **Request Parameters**:
    - `question` (form parameter): The user's research question.
- **Response**: HTML rendering of the research results.

## Examples

### Performing a Web Research Task

To perform a web-based research task, navigate to `http://localhost:8000/` in your browser. Enter your research question and click the "Submit" button. The server will provide the results in HTML format.

### Generating a Research Report

To generate a detailed research report, use either the ArXiv Research Assistant endpoint (`/arxiv-research-assistant`) or the Web Research Assistant endpoint (`/web-research-assistant`). Submit a research question using the form, and the server will generate a report in markdown format.

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests. Make sure to follow the contribution guidelines.

## License

This project is licensed under the [MIT License](LICENSE).
