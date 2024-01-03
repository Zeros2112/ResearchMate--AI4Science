from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import json
from dotenv import load_dotenv, find_dotenv
import os 
import openai
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from markdown2 import markdown
from fastapi.staticfiles import StaticFiles


_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')


# This is for Arxiv

from langchain.retrievers import ArxivRetriever

retriever = ArxivRetriever()
SUMMARY_TEMPLATE = """{doc} 

-----------

Using the above text, answer in short the following question: 

> {question}

-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary =  SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
    docs = lambda x: retriever.get_summaries_as_docs(x["question"])
)| (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()



SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501


# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

# Mount "static" directory for serving CSS and other static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates
templates = Jinja2Templates(directory="templates")

# Serve HTML files from the "templates" directory
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/arxiv-research-assistant", response_class=HTMLResponse)
async def web_research_assistant(request: Request, question: str = Form(...)):
    form_data = await request.form()
    question = form_data.get("question")

    results = chain.invoke({"question": f"{question}"})
    print("Results:", results)  # Add this line for debugging

    # Convert line breaks to HTML line breaks
    formatted_results = markdown(results)

    return templates.TemplateResponse(
        "results.html", {"request": request, "results": formatted_results}
    )



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)