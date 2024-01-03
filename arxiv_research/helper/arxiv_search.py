# Importing necessary modules and classes from the 'app' module.
from app import *

# Creating an instance of the ArxivRetriever class for retrieving information from ArXiv.
retriever = ArxivRetriever()

# Defining a template for document summarization, including a placeholder for the document text and the question.
SUMMARY_TEMPLATE = """{doc} 

-----------

Using the above text, answer in short the following question: 

> {question}

-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501

# Creating a ChatPromptTemplate using the defined SUMMARY_TEMPLATE.
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

# Creating a chain for scraping and summarizing documents using the ChatOpenAI model.
# The result is formatted as a string containing the document title and the generated summary.
scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary=SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: f"Title: {x['doc'].metadata['Title']}\n\nSUMMARY: {x['summary']}")

# Creating a chain for web searching based on user-provided questions, retrieving document summaries using ArXivRetriever,
# and then scraping and summarizing the documents using the previously defined chain.
web_search_chain = RunnablePassthrough.assign(
    docs=lambda x: retriever.get_summaries_as_docs(x["question"])
) | (lambda x: [{"question": x["question"], "doc": u} for u in x["docs"]]) | scrape_and_summarize_chain.map()
