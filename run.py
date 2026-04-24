from agent import agent
import uuid

print("Enter 'quit' to exit.")
thread_id = str(uuid.uuid4())

while True:
    question = input("Ask: ")
    if question.strip().lower() in ['quit', 'exit']:
        break
    
    answer = agent(question, file_name=None, thread_id=thread_id)
    print(f"\n{answer}\n")
