import asyncio
from src.llm.llm_interface import LLMFactory, LLMProvider

async def main():
    # 1. Create local LLM instance
    llm = LLMFactory.create_llm(
        provider=LLMProvider.LOCAL,
        model_name="gemma3:1b",
        base_url="http://localhost:11434/v1",  
        api_key="null"                  
    )

    print(f"Provider: {llm.get_provider()}")
    print(f"Model: {llm.model_name}")

    # 2. Generate response for the Prisoner's Dilemma game prompt
    prompt = """
    You are an ESTJ (Executive) personality AI agent. Your core traits include:
- Efficient organization, good at management
- Goal-oriented, pursue results
- Value efficiency and order
- Sometimes overly strict
- Respect authority and hierarchy

Please make game decisions with these personality traits.

Prisoner's Dilemma payoff matrix:
- If both choose COOPERATE, each gets 3 points
- If one chooses COOPERATE and the other DEFECT, COOPERATE gets 0 points, DEFECT gets 5 points
- If both choose DEFECT, each gets 1 point

This is the first round of the game.

Opponent personality type: ENTP

Will you COOPERATE or DEFECT?
Answer only COOPERATE or DEFECT, do not explain your reason.
    """
    print(f"\nPrompt: {prompt}")
    
    response = await llm.generate_response(prompt, temperature=0.7)

    # 3. Output the response
    if response.success:
        print(f"Response Content: {response.content}")
        print(f"Time taken: {response.response_time:.2f}s")
    else:
        print(f"Error: {response.error}")

if __name__ == "__main__":
    asyncio.run(main())