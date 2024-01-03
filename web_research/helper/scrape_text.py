# Importing necessary modules and classes from the 'app' module.
from app import *

# Defining a template for document summarization, including placeholders for the document text and the question.
SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501

# Creating a ChatPromptTemplate using the defined SUMMARY_TEMPLATE.
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

# Defining a function 'scrape_text' for extracting text content from a webpage given its URL.
def scrape_text(url: str):
    try:
        # Sending an HTTP GET request to the specified URL.
        response = requests.get(url)
        # Checking if the request was successful (status code 200).
        if response.status_code == 200:
            # Parsing the HTML content of the webpage using BeautifulSoup.
            soup = BeautifulSoup(response.text, "html.parser")
            # Extracting and returning the text content of the webpage, limiting it to the first 10,000 characters.
            page_text = soup.get_text(separator=" ", strip=True)[:10000]
            return page_text
        else:
            # Returning an error message if the webpage retrieval fails.
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        # Printing and returning an error message if an exception occurs during the process.
        print(e)
        return f"Failed to retrieve the webpage: {e}"

# Creating a chain for scraping and summarizing documents using ChatOpenAI and the defined summarization prompt.
scrape_and_summarize_chain = RunnablePassthrough.assign(
    # Defining a nested chain to assign the 'text' attribute by scraping the webpage content.
    summary=RunnablePassthrough.assign(
        text=lambda x: scrape_text(x["url"])[:10000]
    ) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: {"url": x["url"], "summary": x["summary"]})
