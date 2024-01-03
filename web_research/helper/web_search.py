# Importing necessary modules and classes from the 'app' module.
from app import *

# Defining a function 'web_search' that performs a web search using DuckDuckGo Search API and returns a list of result links.
# The function takes a query string and an optional parameter 'num_results' specifying the number of results to retrieve.
def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    # Using the 'results' method of the 'ddg_search' object to obtain search results.
    results = ddg_search.results(query, num_results)
    # Extracting and returning the links from the results.
    return [r["link"] for r in results]

# Creating a chain for web searching based on user-provided questions, using the 'web_search' function.
# The chain creates a list of dictionaries containing the question and corresponding URL.
# It then performs scraping and summarization of the documents using the 'scrape_and_summarize_chain'.
web_search_chain = RunnablePassthrough.assign(
    # Defining a lambda function to call 'web_search' with the user's question and obtain a list of URLs.
    urls=lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()
