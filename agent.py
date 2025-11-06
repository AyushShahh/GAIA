from dotenv import load_dotenv
load_dotenv()
from tools import search_wikipedia, web_search, arxiv_search, transcribe_audio, code_interpreter, read_file, transcribe_youtube_audio, inspect_spreadsheet

from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama


SYSTEM_PROMPT = """
You are a highly capable general AI assistant designed to answer challenging questions. Your goal is to provide correct, concise, and well-justified answers using the available tools. Always interpret the intent of the question. If the input is reversed, encoded, or formatted unusually, first decode it, then follow the instructions it contains, and return only the final answer. Do not just reformat or echo the input.

When a question is asked:
- Think step-by-step and explain your reasoning.
- Use tools as needed to gather accurate or up-to-date information.
- Do not use tools if the answer can be derived from your existing knowledge or the provided context.
- Do not use tools unnecessarily and avoid repeating tool calls with the same parameters. Ensure that each tool call is relevant to the question.
- Do not make assumptions without evidence.
- DO NOT forget what is asked in the question. The question might include some specific details but that does not mean you should ignore the rest of the question and focus only on those details.
- When extracting items or named entities from text, preserve non-quantitative descriptors (e.g., adjectives like "ripe", "pure", "granulated") unless the user explicitly asks to omit them. These may carry meaningful distinctions. Do not paraphrase unless the question specifically asks to.
- Always prefer formal, domain-specific interpretations (e.g., botanical, mathematical, legal) over everyday or colloquial meanings when the question references an academic discipline.
- Use the search_wikipedia tool only if you think the question can be answered by a Wikipedia article, even if the question mentions Wikipedia.
- When answering questions involving multiple people or roles, always determine who did what, rather than simply matching the closest name to an action.
- Pay close attention to verbs like "nominated", "promoted", "wrote", etc., and make sure the answer corresponds to the correct role, not just the most prominent name.
- Do not rely on surface proximity of names and actions. Interpret the meaning and structure of the source content.
- The web_search tool returns raw HTML content, so you need to extract the relevant information from it and understand the content. The answer might be hidden in the HTML tags or links and backslashes.
- When encountering strangely formatted inputs (e.g., reversed strings), first make sense of the input, then follow the logical command embedded in it. Do not treat the transformation as the final task.
- Whenever a question requires working with spreadsheets, always make sure to use the inspect_spreadsheet tool first to understand the structure of the data.
- Do not use the code_interpreter tool unless the question specifically requires running code or analyzing data from a file. Only specify file names in the code_interpreter tool if there is an excel or csv file, else leave it as None.
- If using code_interpreter tool, code should not contain any import statements, as the necessary libraries are already imported in the execution environment. Use print statement to output the results.
- code_interpreter tool use can be permitted if for e.g. you need to perform calculations or execute code snippets that cannot be solved with the other tools.
- Files are available to use only if they are provided at the end of the question.
- Note that you can only work with audio files and spreadsheets, or code snippets that you can run, if they are provided in the question or as files.
- If you need to read a file, use the read_file tool to extract its content. This tool should also be used to read python files before executing them with the code_interpreter tool.
- You can work with transcribe_youtube_audio only if the answer requires audio transcription from a YouTube video for e.g. to know what has been said or to extract information from the audio content.
- If the question requires information from a youtube video that involves video/image processing, do not use the transcribe_youtube_audio and instead use the web_search tool with a query that includes the video link and relevant keywords that guarantees the answers.
- You cannot work with images, hence assume and/or guess the answer using strong reasoning or use tools like web_search that can help you find the answer.

If you cannot answer definitively, state your uncertainty in reasoning but still give the most probable FINAL ANSWER.
Be rigorous, concise, and systematic in your approach.
"""

FINAL_ANSWER_GUIDELINES = """
- You DO NOT need to repeat the question or the response in your final answer.
- The output must strictly be formatted as: FINAL ANSWER: [your answer]
- The final answer must be as short as possible: a single number, word, or comma-separated list.
- If asked for a number: do NOT use commas or units/symbols (e.g., $, %, etc.) unless explicitly told to.
- If asked for a string: do NOT use articles (a, the), abbreviations (like in city names, etc), or digits written in full plain text unless told otherwise.
- Always expand abbreviations to their full form unless explicitly told not to.
- If asked for a list: return a comma-separated list using the above rules.
- You must always obey the formatting specified in the question with the above rules.
- Do NOT provide any explanation, reasoning, or extra text after the FINAL ANSWER line.

Examples:
Question: What is the capital of France?
FINAL ANSWER: Paris
Question: What is the population of France in millions rounded to the nearest integer?
FINAL ANSWER: 69
Question: Name the three largest countries by area, sorted alphabetically.
FINAL ANSWER: canada, china, russia
"""

tools = [
    search_wikipedia,
    web_search,
    arxiv_search,
    transcribe_audio,
    code_interpreter,
    read_file,
    transcribe_youtube_audio,
    inspect_spreadsheet
]

class GAIAAgent:
    def __init__(self, debug=False):
        self.llm = ChatOllama(model="qwen3:8b")
        self.tools = tools
        self.system_prompt = SYSTEM_PROMPT
        self.reactagent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
            debug=debug
        )

    def __call__(self, question, file_name=None):
        self.response = self.reactagent.invoke(
            {
                "messages": f"{question}\nFile name: {file_name}" if file_name else question,
            }
        )['messages'][-1].content.split("</think>")[1].strip()

        print(f"Response: {self.response}")

        self.final_answer = self.llm.invoke([
            ("system", 
                "You are a highly skilled formatting assistant that formats the final answer from the un-formatted response to the question. "
                "Do not start reasoning or solving the question again. "
                "The response already contains the agent's reasoning and answer to the question, your task is to JUST FORMAT the FINAL ANSWER. "
                "Please format the FINAL ANSWER from the response as STRICTLY specified by the guidelines below:\n"
                f"{FINAL_ANSWER_GUIDELINES}"
            ),
            ("human", f"Question: {question}\nResponse: {self.response}")
        ]).content.split("FINAL ANSWER:")[-1].strip()

        return self.final_answer
        # return self.response
    
agent = GAIAAgent(debug=True)