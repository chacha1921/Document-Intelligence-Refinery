
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

def test_gemini_tool_history_v2():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Skipping test, no API key")
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )

    # Mimic the exact failure state
    messages = [
        HumanMessage(content="What caused the massive increase in excise duty tax expenditures for motor vehicles between 2019/20 and 2020/21, and what would the total tax expenditure for vehicles fall to if the benchmark excise rate was capped at 100%?"),
        AIMessage(
            content="", 
            tool_calls=[{
                "name": "semantic_search", 
                "args": {"query": "excise duty tax expenditures motor vehicles 2019/20 2020/21 increase reason"}, 
                "id": "call_unique_id_1"
            }]
        ),
        ToolMessage(
            content="Error: Vector database not found at .refinery/chroma_db. Has the indexing (Stage 3) been run?", 
            tool_call_id="call_unique_id_1"
        )
    ]

    print("Attempting to invoke LLM with failure state messages...")
    try:
        response = llm.invoke(messages)
        print("Success!")
        print(response)
    except Exception as e:
        print("\n--- FAILED ---")
        print(e)
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    test_gemini_tool_history_v2()
