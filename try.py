import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from uagents import Agent, Context
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from fetchai.registration import register_with_agentverse
from fetchai.crypto import Identity


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

class TickerRequest(BaseModel):
    ticker: str

class FinancialNewsSentimentRequest(BaseModel):
    ticker: str

class NewsSentiment(BaseModel):
    title: str
    url: str
    summary: str
    overall_sentiment_label: str

class FinancialNewsSentimentResponse(BaseModel):
    summary: List[NewsSentiment]

AI_AGENT_ADDRESS = "agent1qdcnxjrr5u5jkqqtcaeqdxxpxne47nvcrm4k3krsprwwgnx50hg96txxjuf"

agent = Agent(name="user", endpoint="http://localhost:8001/submit")

# Function to initialize the client agent
def init_client():
    try:
        client_identity = Identity.from_seed("Unique String", 0)
        readme = """
        <description>Frontend client that tests with Stocks price using alphavantage.</description>
        """
        register_with_agentverse(
            identity=client_identity,
            url="http://localhost:8000/analyze_company",
            agentverse_token=os.getenv("AGENTVERSE_TOKEN"),
            agent_title="Stock Analysis Agent",
            readme=readme
        )
        print("Client agent registration complete!")
    except Exception as e:
        print(f"Initialization error: {e}")
        raise

@app.post("/analyze_company")
async def analyze_company(request: TickerRequest) -> Dict[str, Any]:
    try:
        loader = WebBaseLoader(f"https://finance.yahoo.com/quote/{request.ticker}")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="deepseek-r1-distill-qwen-32b"
        )
        prompt = ChatPromptTemplate.from_template("""
        You are a financial expert analyzing company data...
        """)
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": "Analyze the financial condition of the company"})
        try:
            result_text = response["answer"]
            result = json.loads(result_text)
            return {"response": result}
        except json.JSONDecodeError:
            return {"response": response["answer"], "note": "Could not parse response as JSON"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing company: {str(e)}")

async def get_sentiment_analysis(ctx: Context, ticker: str) -> Dict[str, Any]:
    await ctx.send(AI_AGENT_ADDRESS, {"ticker": ticker})
    ctx.logger.info(f"Sent sentiment request for {ticker}")
    return await ctx.receive()

@app.post("/combined_analysis")
async def combined_analysis(request: TickerRequest) -> Dict[str, Any]:
    try:
        ticker = request.ticker
        financial_analysis = await analyze_company(request)
        sentiment_analysis = await get_sentiment_analysis(agent.context, ticker)
        return {
            "ticker": ticker,
            "financial_analysis": financial_analysis,
            "sentiment_analysis": sentiment_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in combined analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    init_client()
    uvicorn.run(app, host="0.0.0.0", port=8000)

