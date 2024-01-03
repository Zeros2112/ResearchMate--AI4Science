# Importing necessary modules and classes from the 'app' module.
from app import *

# Defining a template for research reports, specifying the structure, depth, and requirements for the report.
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
Write all used source URLs at the end of the report, and make sure not to add duplicated sources, but only one reference for each.
You must write the report in APA format.
Please do your best; this is very important to my career."""  # noqa: E501

# Creating a ChatPromptTemplate using the defined RESEARCH_REPORT_TEMPLATE.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

# Defining a function 'collapse_list_of_lists' to format a list of lists into a structured string.
def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        if isinstance(l, (dict, list)):
            content.append("\n\n".join(map(str, l)))
        else:
            content.append(str([l]))
    return "\n\n".join(content)

# Creating a chain for generating research reports, including collapsing the list of lists obtained from full_research_chain.
chain = RunnablePassthrough.assign(
    research_summary=full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
