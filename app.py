from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

print("i am your chartbot")

System_insstruction = SystemMessage(content="You are a helpful assistant that can create charts based on user input. You can use the following tools to create charts: \n\n1. Line Chart: Use this tool to create a line chart. Provide the data points and labels for the x and y axes.\n2. Bar Chart: Use this tool to create a bar chart. Provide the categories and their corresponding values.\n3. Pie Chart: Use this tool to create a pie chart. Provide the categories and their corresponding values.\n4. Scatter Plot: Use this tool to create a scatter plot. Provide the data points and labels for the x and y axes.\n5. Histogram: Use this tool to create a histogram. Provide the data points and specify the number of bins.\n\nWhen responding to user input, analyze the request and determine which chart type is most appropriate based on the provided data and context. Then, use the corresponding tool to generate the chart.")

model = ChatOllama(model="llama3.2:3b", temperature=0.7)

print("Exit to quit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messages = [
        System_insstruction,
        HumanMessage(content=user_input)
    ]

    response = model.invoke(messages)
    print(f"ChartBot: {response.content}")