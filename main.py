from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

console = Console()


@tool
def get_price(city: str, date: str) -> str:
    """Получить цену/погоду для заданного города и даты."""
    return f"Цена в {city} на {date}: 100 рублей"


memory = MemorySaver()

agent = create_agent(
    model="openai/gpt-4o-mini",
    tools=[get_price],
    system_prompt="Ты полезный ассистент. Отвечай на русском языке.",
    checkpointer=memory,
    interrupt_before=["tools"],
)

config = {"configurable": {"thread_id": "session-1"}}


def ask_and_run(user_input, config):
    for chunk in agent.stream(
        user_input, config=config, stream_mode=["messages", "updates"]
    ):
        state = agent.get_state(config)
        chunk_type, chunk_data = chunk

        if chunk_type == "messages":
            for msg in chunk_data:
                if hasattr(msg, "content") and msg.content:
                    console.print(msg.content, end="")

        if chunk_type == "updates":
            for _node, updates in chunk_data.items():
                if "messages" in updates:
                    for msg in updates["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                console.print(
                                    f"\n --- --- --- \n{tc['name']}({tc['args']})"
                                )

        if "__interrupt__" in chunk_data and state.next == ("tools",):
            tool_call = state.values["messages"][-1].tool_calls[0]
            console.print(
                f"Агент хочет вызвать утилиту {tool_call['name']}({tool_call['args']})"
            )
            answer = input("Разрешить? (Y/n): ")

            if answer.lower().strip() == "y":
                ask_and_run(None, config)
            else:
                console.print("Отменено")
                break


def main():
    while True:
        user_input = input("\nВы: ")
        if user_input == "exit":
            break

        ask_and_run(
            {"messages": [{"role": "human", "content": user_input}]},
            config,
        )


if __name__ == "__main__":
    main()
