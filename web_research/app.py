from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import json
from dotenv import load_dotenv, find_dotenv
import os
import requests
from bs4 import BeautifulSoup
import openai
from langserve import add_routes
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from markdown2 import markdown


_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

RESULTS_PER_QUESTION = 5

ddg_search = DuckDuckGoSearchAPIWrapper()

def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]

SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

def scrape_text(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"

url = "https://blog.langchain.dev/announcing-langsmith/"

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary=RunnablePassthrough.assign(
        text=lambda x: scrape_text(x["url"])[:10000]
    ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: {"url": x["url"], "summary": x["summary"]})

web_search_chain = RunnablePassthrough.assign(
    urls=lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

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

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well-written, critically acclaimed, objective and structured reports on given text."  # noqa: E501

RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well-structured, informative, \
in-depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in APA format.
Please do your best; this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)
def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        if isinstance(l, (dict, list)):
            content.append("\n\n".join(map(str, l)))
        else:
            content.append(str([l]))
    return "\n\n".join(content)



chain = RunnablePassthrough.assign(
    research_summary=full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

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

@app.post("/web-research-assistant", response_class=HTMLResponse)
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
