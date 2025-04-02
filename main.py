import os
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import logging
import asyncio
from threading import Thread
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class UrlRequest(BaseModel):
    url: str
    
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

class CombinedAnalysisRequest(BaseModel):
    ticker: str

class CombinedAnalysisResponse(BaseModel):
    financial_analysis: Dict[str, Any]
    sentiment_analysis: List[NewsSentiment]
    combined_explanation: str

financial_agent = Agent(
    name="financial-analysis-agent",
    port=8001,
    endpoint=["http://127.0.0.1:8001/submit"],
    seed="yojmiwjcmwkekrase"
)

news_sentiment_agent = Agent(
    name="news-sentiment-agent",
    port=8002,
    endpoint=["http://127.0.0.1:8002/submit"],
    seed="newssentimentagentseedphrase"
)

coordinator_agent = Agent(
    name="coordinator-agent",
    port=8003,
    endpoint=["http://127.0.0.1:8003/submit"],
    seed="coordinatoragentseedphrase"
)

fund_agent_if_low(financial_agent.wallet.address())
fund_agent_if_low(news_sentiment_agent.wallet.address())
fund_agent_if_low(coordinator_agent.wallet.address())

financial_protocol = Protocol("Financial Analysis")
news_sentiment_protocol = Protocol("News Sentiment Analysis")
coordinator_protocol = Protocol("Analysis Coordinator")

analysis_results = {}

@financial_protocol.on_message(model=FinancialNewsSentimentRequest)
async def handle_financial_request(ctx: Context, sender: str, msg: FinancialNewsSentimentRequest):
    ctx.logger.info(f"Received financial analysis request for: {msg.ticker}")
    
    try:
        financial_results = await analyze_ticker(msg.ticker)
        
        request_id = f"financial_{msg.ticker}_{sender}"
        analysis_results[request_id] = financial_results
        
        await ctx.send(
            sender,
            financial_results
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing financial analysis request: {str(e)}")

        await ctx.send(
            sender,
            {"error": f"Failed to analyze {msg.ticker}: {str(e)}"}
        )

@news_sentiment_protocol.on_message(model=FinancialNewsSentimentRequest)
async def handle_sentiment_request(ctx: Context, sender: str, msg: FinancialNewsSentimentRequest):
    ctx.logger.info(f"Received news sentiment request for: {msg.ticker}")
    
    try:
        sentiment_results = await analyze_news_sentiment(msg.ticker)
        
        request_id = f"sentiment_{msg.ticker}_{sender}"
        analysis_results[request_id] = sentiment_results
        
        await ctx.send(
            sender,
            FinancialNewsSentimentResponse(summary=sentiment_results)
        )
        
    except Exception as e:
        ctx.logger.error(f"Error processing sentiment request: {str(e)}")
    
        await ctx.send(
            sender,
            FinancialNewsSentimentResponse(summary=[
                NewsSentiment(
                    title="Error",
                    url="",
                    summary=f"Failed to analyze sentiment for {msg.ticker}: {str(e)}",
                    overall_sentiment_label="neutral"
                )
            ])
        )

@coordinator_protocol.on_message(model=CombinedAnalysisRequest)
async def handle_combined_analysis_request(ctx: Context, sender: str, msg: CombinedAnalysisRequest):
    ctx.logger.info(f"Received combined analysis request for: {msg.ticker}")
    
    try:
        analysis_id = f"combined_{msg.ticker}_{sender}"
        ctx.logger.info(f"Requesting financial analysis for {msg.ticker}")
        await ctx.send(
            financial_agent.address,
            FinancialNewsSentimentRequest(ticker=msg.ticker)
        )
        
        ctx.logger.info(f"Requesting news sentiment analysis for {msg.ticker}")
        await ctx.send(
            news_sentiment_agent.address,
            FinancialNewsSentimentRequest(ticker=msg.ticker)
        )
        
        financial_results = None
        sentiment_results = None
        
        for _ in range(10):  
            await asyncio.sleep(2)  
            financial_key = f"financial_{msg.ticker}_{financial_agent.address}"
            sentiment_key = f"sentiment_{msg.ticker}_{news_sentiment_agent.address}"
            
            if financial_key in analysis_results:
                financial_results = analysis_results[financial_key]
            
            if sentiment_key in analysis_results:
                sentiment_results = analysis_results[sentiment_key]
            
            if financial_results and sentiment_results:
                break
        
        if financial_results and sentiment_results:
            combined_explanation = await generate_combined_explanation(
                msg.ticker, financial_results, sentiment_results
            )
            
            response = CombinedAnalysisResponse(
                financial_analysis=financial_results,
                sentiment_analysis=sentiment_results,
                combined_explanation=combined_explanation
            )
            
            analysis_results[analysis_id] = response
            await ctx.send(sender, response)
        else:
            raise Exception("Timed out waiting for analysis results")
            
    except Exception as e:
        ctx.logger.error(f"Error processing combined analysis request: {str(e)}")
    
        await ctx.send(
            sender,
            {"error": f"Failed to perform combined analysis for {msg.ticker}: {str(e)}"}
        )

financial_agent.include(financial_protocol)
news_sentiment_agent.include(news_sentiment_protocol)
coordinator_agent.include(coordinator_protocol)

async def analyze_ticker(ticker: str, url: str = None):
    """Analyze a company based on ticker and optional URL"""
    try:
        if not url:
            url = f"https://finance.yahoo.com/quote/{ticker}"
        
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        llm = ChatGroq(
            temperature=0, 
            groq_api_key="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", 
            model_name="deepseek-r1-distill-qwen-32b"
        )
        
        prompt = ChatPromptTemplate.from_template("""
        You are a financial expert analyzing company data for ticker {ticker}.
        Based on the following information, provide a comprehensive financial analysis of the company.
        
        Context information:
        {context}
        
        Analyze the company's financial condition and provide a structured response with exactly these four sections:
        1. NSE: Provide the National Stock Exchange (NSE) code of the company.
        2. Sales Growth: Analyze the company's sales growth patterns and performance.
        3. Profit Growth: Evaluate the company's profit growth trends and performance.
        4. ROE: Explain the Return on Equity percentage and what it indicates about the company.
        5. Market Analysis: Provide detailed analysis of the company's market position and trends.
        
        Format your response as a valid JSON object with these exact keys: "NSE","market_analysis", "sales_growth", "profit_growth", "roe" with string values for each section.
        The response should be valid JSON that can be parsed by Python's json.loads() function.
        """)
        
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({"input": f"Analyze the financial condition of {ticker}", "ticker": ticker})
        
        try:
            result_text = response["answer"]
            
            if "```json" in result_text:
                json_content = result_text.split("```json")[1].split("```")[0].strip()
                result = json.loads(json_content)
            elif "```" in result_text:
                json_content = result_text.split("```")[1].split("```")[0].strip()
                result = json.loads(json_content)
            else:
                result = json.loads(result_text)
                
            return result
        except json.JSONDecodeError:
            return {
                "NSE": ticker,
                "market_analysis": response["answer"],
                "sales_growth": "Could not parse",
                "profit_growth": "Could not parse",
                "roe": "Could not parse"
            }
    
    except Exception as e:
        logger.error(f"Error analyzing company: {str(e)}")
        raise

async def analyze_news_sentiment(ticker: str):
    """Analyze news sentiment for a company based on ticker using a news API"""
    try:
        # NewsAPI approach
        news_api_key = "5a2495c1-272e-43c5-a49b-910c8f3bfc26"
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={ticker}+OR+{ticker.split('.')[0]}+stock&"  # Include both ticker and company name
            f"language=en&"
            f"sortBy=relevancy&"
            f"apiKey={news_api_key}&"
            f"pageSize=5&"
            f"domains=bloomberg.com,reuters.com,marketwatch.com,finance.yahoo.com"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                
        news_items = []
        if data.get("status") == "ok" and data.get("articles"):
            for article in data.get("articles")[:3]:
                title = article.get("title", "Unknown")
                url = article.get("url", "")
                summary = article.get("description", "No summary available")
                
                # Use LLM to analyze sentiment
                llm = ChatGroq(
                    temperature=0,
                    groq_api_key="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",
                    model_name="deepseek-r1-distill-qwen-32b"
                )
                
                sentiment_prompt = f"""
                Analyze the sentiment of this news about {ticker} stock:
                Title: {title}
                Summary: {summary}
                
                Reply with just one word: 'positive', 'negative', or 'neutral'.
                """
                
                sentiment_response = await llm.ainvoke(sentiment_prompt)
                sentiment = sentiment_response.content.strip().lower()
                
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"
                
                news_items.append(NewsSentiment(
                    title=title,
                    url=url,
                    summary=summary,
                    overall_sentiment_label=sentiment
                ))
                
            return news_items
        else:
            return [
                NewsSentiment(
                    title=f"{ticker} Recent Performance",
                    url=f"https://finance.yahoo.com/quote/{ticker}",
                    summary=f"Visit Yahoo Finance for the latest updates on {ticker}.",
                    overall_sentiment_label="neutral"
                )
            ]
                
    except Exception as e:
        logger.error(f"Error analyzing news sentiment: {str(e)}")
        return [
            NewsSentiment(
                title=f"{ticker} Market Update",
                url="",
                summary=f"Could not retrieve news at this time for {ticker}.",
                overall_sentiment_label="neutral"
            )
        ]
    
async def generate_combined_explanation(ticker: str, financial_results: Dict[str, Any], sentiment_results: List[NewsSentiment]):
    """Generate combined explanation of financial and sentiment analysis"""
    try:
        llm = ChatGroq(
            temperature=0.2,  
            groq_api_key="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", 
            model_name="deepseek-r1-distill-qwen-32b"
        )
        
        sentiment_summaries = []
        overall_sentiment = "neutral"
        
        for item in sentiment_results:
            sentiment_summaries.append(f"- {item.title}: {item.overall_sentiment_label} - {item.summary}")
            if item.overall_sentiment_label == "positive":
                overall_sentiment = "positive"
            elif item.overall_sentiment_label == "negative" and overall_sentiment != "positive":
                overall_sentiment = "negative"
        
        sentiment_text = "\n".join(sentiment_summaries)
        
        prompt = f"""
        You are a financial advisor analyzing both financial data and news sentiment for {ticker}.
        
        FINANCIAL ANALYSIS:
        - Market Analysis: {financial_results.get('market_analysis', 'Not available')}
        - Sales Growth: {financial_results.get('sales_growth', 'Not available')}
        - Profit Growth: {financial_results.get('profit_growth', 'Not available')}
        - ROE: {financial_results.get('roe', 'Not available')}
        
        NEWS SENTIMENT ANALYSIS:
        {sentiment_text}
        Overall sentiment: {overall_sentiment}
        
        Please provide a comprehensive explanation of how the financial data and news sentiment relate to each other.
        Analyze areas where they align or contradict, and what this might indicate about the company's current position
        and future prospects. Be specific about the implications for investors, providing actionable insights based on 
        both quantitative financial metrics and qualitative news sentiment.
        
        Your explanation should be between 3-5 paragraphs and focus on the most important connections between the 
        financial performance and market perception.
        """
        
        response = await llm.ainvoke(prompt)
        combined_explanation = response.content
        
        return combined_explanation
    
    except Exception as e:
        logger.error(f"Error generating combined explanation: {str(e)}")
        return f"Unable to generate combined explanation: {str(e)}"

@app.post("/analyze_company")
async def analyze_company(request: UrlRequest) -> Dict[str, Any]:
    """FastAPI endpoint for analyzing a company based on URL"""
    try:
        ticker = request.url.split("/")[-1]
        if "=" in ticker:
            ticker = ticker.split("=")[-1]
            
        result = await analyze_ticker(ticker, request.url)
        return {"response": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing company: {str(e)}")

@app.post("/analyze_news_sentiment")
async def analyze_news_sentiment_endpoint(request: TickerRequest) -> Dict[str, Any]:
    """FastAPI endpoint for analyzing news sentiment for a ticker"""
    try:
        result = await analyze_news_sentiment(request.ticker)
        return {"response": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing news sentiment: {str(e)}")

@app.post("/combined_analysis")
async def combined_analysis_endpoint(request: TickerRequest) -> Dict[str, Any]:
    """FastAPI endpoint for performing combined financial and sentiment analysis"""
    try:
        financial_result = await analyze_ticker(request.ticker)
        
        sentiment_result = await analyze_news_sentiment(request.ticker)
        
        explanation = await generate_combined_explanation(
            request.ticker, financial_result, sentiment_result
        )
        
        return {
            "financial_analysis": financial_result,
            "sentiment_analysis": sentiment_result,
            "combined_explanation": explanation
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing combined analysis: {str(e)}")

@app.get("/example_client")
async def example_client():
    """Returns example code for a client to communicate with the coordinator agent"""
    example_code = """
from uagents import Agent, Context, Model
from typing import Dict, Any, List

class CombinedAnalysisRequest(Model):
    ticker: str

class NewsSentiment(Model):
    title: str
    url: str
    summary: str
    overall_sentiment_label: str

class CombinedAnalysisResponse(Model):
    financial_analysis: Dict[str, Any]
    sentiment_analysis: List[NewsSentiment]
    combined_explanation: str

client_agent = Agent(
    name="client-agent",
    seed="unique-client-seed-phrase"
)

COORDINATOR_AGENT_ADDRESS = "agent1q..." # Replace with actual coordinator agent address
ticker = "AAPL"  # Replace with desired ticker

@client_agent.on_event("startup")
async def send_request(ctx: Context):
    await ctx.send(COORDINATOR_AGENT_ADDRESS, CombinedAnalysisRequest(ticker=ticker))
    ctx.logger.info(f"Sent combined analysis request for ticker: {ticker}")

@client_agent.on_message(CombinedAnalysisResponse)
async def handle_response(ctx: Context, sender: str, msg: CombinedAnalysisResponse):
    ctx.logger.info(f"Received combined analysis from {sender}:")
    
    # Financial Analysis
    ctx.logger.info("=== FINANCIAL ANALYSIS ===")
    for key, value in msg.financial_analysis.items():
        ctx.logger.info(f"{key}: {value}")
    
    # News Sentiment
    ctx.logger.info("=== NEWS SENTIMENT ===")
    for news in msg.sentiment_analysis:
        ctx.logger.info(f"Title: {news.title}")
        ctx.logger.info(f"URL: {news.url}")
        ctx.logger.info(f"Summary: {news.summary}")
        ctx.logger.info(f"Sentiment: {news.overall_sentiment_label}")
    
    # Combined Explanation
    ctx.logger.info("=== COMBINED EXPLANATION ===")
    ctx.logger.info(msg.combined_explanation)

if __name__ == "__main__":
    client_agent.run()
"""
    return {"example_code": example_code}

def run_agent(agent_obj):
    """Function to run a uAgent in a separate thread"""
    agent_obj.run()

# Run the app with all agents
if __name__ == "__main__":
    import uvicorn
    
    # Start agents in separate threads
    financial_thread = Thread(target=run_agent, args=(financial_agent,))
    financial_thread.daemon = True
    financial_thread.start()
    
    sentiment_thread = Thread(target=run_agent, args=(news_sentiment_agent,))
    sentiment_thread.daemon = True
    sentiment_thread.start()
    
    coordinator_thread = Thread(target=run_agent, args=(coordinator_agent,))
    coordinator_thread.daemon = True
    coordinator_thread.start()
    
    # Start FastAPI in the main thread
    uvicorn.run(app, host="0.0.0.0", port=8000)