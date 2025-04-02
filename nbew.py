from typing import List
from uagents import Agent, Context, Model


class FinancialNewsSentimentRequest(Model):
    ticker: str


class NewsSentiment(Model):
    title: str
    url: str
    summary: str
    overall_sentiment_label: str


class FinancialNewsSentimentResponse(Model):
    summary: List[NewsSentiment]
agent = Agent(
    name="user",
    endpoint="http://localhost:8001/submit",
)


AI_AGENT_ADDRESS = "agent1qdcnxjrr5u5jkqqtcaeqdxxpxne47nvcrm4k3krsprwwgnx50hg96txxjuf"

ticker = "AAPL"


@agent.on_event("startup")
async def send_message(ctx: Context):
    await ctx.send(AI_AGENT_ADDRESS, FinancialNewsSentimentRequest(ticker=ticker))
    ctx.logger.info(f"Sent prompt to AI agent: {ticker}")


@agent.on_message(FinancialNewsSentimentResponse)
async def handle_response(ctx: Context, sender: str, msg: FinancialNewsSentimentResponse):
    ctx.logger.info(f"Received response from {sender}:")
    ctx.logger.info(msg.summary)


if __name__ == "__main__":
    agent.run()
    
    
#output

