# Importing necessary modules and classes from the 'app' module.
from app import *

# Creating a search prompt using the ChatPromptTemplate class, instructing the user to provide 3 Google search queries.
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

# Creating a chain of operations for searching questions using OpenAI's ChatOpenAI model, parsing the output, and loading it as JSON.
# The temperature parameter is set to 0 for deterministic responses.
search_question_chain = SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads

# Creating a full research chain that takes user-provided questions, converts them into a list of dictionaries,
# and then performs web searches for each question using the 'web_search_chain'.
full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()
