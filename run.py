from agent import agent

question = input("Ask: ")
answer = agent(question, file_name=None)
print(f"\n{answer}")