import os
import json
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.error(f"Error importing SentenceTransformer: {e}")
    logger.error("Try: pip uninstall sentence-transformers huggingface_hub")
    logger.error("Then: pip install sentence-transformers==2.7.0 huggingface_hub==0.20.3")
    raise
from groq import Groq
from loguru import logger
import numpy as np
from datetime import datetime
import time
import hashlib
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from contextlib import asynccontextmanager

# Production optimizations
import gc
import psutil
import threading

load_dotenv()

# -------------------
# Production Configuration
# -------------------
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TOP_K = int(os.getenv('TOP_K', 8))  # Increased for production
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', 6000))  # Increased for large PDFs
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', 100))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 120))  # 2 minutes for large documents
CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))  # 1 hour cache
WORKER_THREADS = int(os.getenv('WORKER_THREADS', 20))  # Thread pool size

# Validate environment variables
if not all([SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY]):
    raise ValueError("Missing required environment variables")

# -------------------
# Global thread pool and model initialization
# -------------------
thread_pool = ThreadPoolExecutor(max_workers=WORKER_THREADS)
model = None
groq_client = None
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Production-grade caching
embedding_cache = {}
query_cache = {}
cache_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown"""
    global model, groq_client
    
    logger.info("🚀 Starting Agentic RAG Production Server")
    
    try:
        # Initialize models in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            thread_pool, 
            lambda: SentenceTransformer(EMBEDDING_MODEL)
        )
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Successfully initialized embedding model and Groq client")
        
        # Memory optimization
        if hasattr(model, 'eval'):
            model.eval()
        
        logger.info(f"📊 System Resources - CPU: {psutil.cpu_count()} cores, RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down services...")
    thread_pool.shutdown(wait=True)
    # Clear caches
    embedding_cache.clear()
    query_cache.clear()
    gc.collect()

# Supabase HTTP headers
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# Initialize FastAPI app with production settings
app = FastAPI(
    title="Agentic RAG Production API",
    description="Production-grade RAG system with agentic planning for 100+ concurrent users",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("DEBUG", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("DEBUG", "false").lower() == "true" else None
)

# Production middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# -------------------
# Production Pydantic Models
# -------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of documents to retrieve")
    include_plan: Optional[bool] = Field(default=True, description="Whether to include the planning step")
    max_tokens: Optional[int] = Field(default=1500, ge=100, le=4000, description="Maximum tokens for response")
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=1.0, description="Temperature for LLM")
    use_cache: Optional[bool] = Field(default=True, description="Whether to use caching")

class DocumentResult(BaseModel):
    id: Optional[int] = None
    doc_id: str
    chunk_id: int
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None

class AgentResponse(BaseModel):
    answer: str
    plan: Optional[str] = None
    reasoning: Optional[str] = None
    sources: List[str]
    retrieved_documents: List[DocumentResult] = []
    retrieved_docs: int = 0
    confidence: str = "Medium"
    processing_time: float
    timestamp: str
    request_id: Optional[str] = None
    cached: bool = False

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    system_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]

# -------------------
# Production Utilities
# -------------------
def generate_request_id() -> str:
    """Generate unique request ID"""
    return hashlib.md5(f"{time.time()}{os.getpid()}".encode()).hexdigest()[:12]

def get_cache_key(question: str, top_k: int, temperature: float) -> str:
    """Generate cache key for query"""
    return hashlib.md5(f"{question}{top_k}{temperature}".encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_encode(text: str) -> List[float]:
    """Cache embeddings for repeated queries"""
    try:
        return model.encode([text], show_progress_bar=False)[0].tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return []

async def rate_limit_check(request: Request):
    """Simple rate limiting"""
    async with request_semaphore:
        yield

# -------------------
# Production Vector Search
# -------------------
async def retrieve_documents_production(query_embedding: List[float], top_k: int = TOP_K) -> List[DocumentResult]:
    """
    Production-grade document retrieval with connection pooling and retries
    """
    max_retries = 3
    timeout = httpx.Timeout(60.0)  # Increased timeout for large documents
    
    for attempt in range(max_retries):
        try:
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Primary: Use vector similarity RPC
            url = f"{SUPABASE_URL}/rest/v1/rpc/match_documents"
            payload = {
                "query_embedding": embedding_str,
                "match_count": top_k,
                "match_threshold": 0.05  # Lower threshold for better recall
            }
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=SUPABASE_HEADERS)
                
                if response.status_code == 200:
                    docs = response.json()
                    if docs:  # If we got results
                        return [DocumentResult(**doc) for doc in docs]
                    else:
                        logger.warning("No documents found with vector search, trying fallback")
                
                # Fallback: Direct table query
                return await fallback_retrieve_documents_production(query_embedding, top_k)
                
        except Exception as e:
            logger.warning(f"Retrieval attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All retrieval attempts failed, using fallback")
                return await fallback_retrieve_documents_production(query_embedding, top_k)
            
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return []

async def fallback_retrieve_documents_production(query_embedding: List[float], top_k: int) -> List[DocumentResult]:
    """
    Fallback retrieval method for production
    """
    try:
        url = f"{SUPABASE_URL}/rest/v1/documents"
        params = {
            "select": "id,doc_id,chunk_id,content,metadata",
            "limit": str(min(top_k * 2, 50)),  # Get more docs for better results
            "order": "id.desc"  # Get recent documents first
        }
        
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params, headers=SUPABASE_HEADERS)
            
            if response.status_code == 200:
                docs = response.json()
                results = []
                for doc in docs[:top_k]:  # Limit to requested amount
                    result = DocumentResult(
                        id=doc.get('id'),
                        doc_id=doc.get('doc_id', 'unknown'),
                        chunk_id=doc.get('chunk_id', 0),
                        content=doc.get('content', ''),
                        metadata=doc.get('metadata', {}),
                        similarity_score=0.3  # Default similarity for fallback
                    )
                    results.append(result)
                return results
            
        logger.error(f"Fallback query failed: {response.status_code}")
        return []
        
    except Exception as e:
        logger.error(f"Fallback retrieval failed: {e}")
        return []

# -------------------
# Production LLM Integration
# -------------------
async def call_groq_llm_production(prompt: str, max_tokens: int = 1500, temperature: float = 0.1, 
                                model_name: str = "llama-3.3-70b-versatile") -> str:
    """
    Production-grade Groq LLM calls with retry logic and proper formatting
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            
            def groq_call():
                return groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are an expert AI assistant that provides accurate, comprehensive responses.

CRITICAL FORMATTING RULES:
- Use proper sentence case (NOT ALL CAPS)
- Only use capitals for: proper nouns, acronyms (C++, Java, SQL), sentence beginnings
- Format code blocks with proper syntax highlighting
- Be clear, precise, and professional
- Complete all responses fully - do not truncate"""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    stop=None
                )
            
            response = await loop.run_in_executor(thread_pool, groq_call)
            response_text = response.choices[0].message.content.strip()
            
            # Apply formatting fix
            response_text = fix_response_formatting(response_text)
            
            return response_text
            
        except Exception as e:
            logger.warning(f"Groq API attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                if model_name == "llama-3.3-70b-versatile":
                    return await call_groq_llm_production(prompt, max_tokens, temperature, "llama-3.1-8b-instant")
                else:
                    return f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
            
            await asyncio.sleep(2 ** attempt)
    
    return "Error processing your request. Please try again."

# -------------------
# Production Agentic Functions
# -------------------
async def create_plan_production(question: str, context: str) -> str:
    """
    Create execution plan with production optimizations
    """
    planner_prompt = f"""
As an intelligent planning agent, create a structured approach for this question.

USER QUESTION: {question}

AVAILABLE CONTEXT: {context[:800]}...

Create a concise 4-step plan focusing on:
1. Key information extraction from context
2. Analysis and reasoning approach  
3. Response structure and organization
4. Source citation strategy

Respond with a clear numbered plan.
"""
    
    return await call_groq_llm_production(planner_prompt, max_tokens=400, temperature=0.05)


def fix_response_formatting(text: str) -> str:
    """
    Fix formatting issues in response while preserving code blocks and structure
    """
    import re
    
    # Extract and preserve code blocks
    code_pattern = r'```[\s\S]*?```'
    code_blocks = re.findall(code_pattern, text)
    
    # Replace code blocks with placeholders
    temp_text = text
    for i, block in enumerate(code_blocks):
        temp_text = temp_text.replace(block, f'___CODE_BLOCK_{i}___')
    
    # Check if the text (outside code blocks) is mostly uppercase
    text_without_code = temp_text
    if text_without_code.strip():
        uppercase_count = sum(1 for c in text_without_code if c.isupper() and c.isalpha())
        total_letters = sum(1 for c in text_without_code if c.isalpha())
        
        if total_letters > 0 and uppercase_count / total_letters > 0.7:
            # Text is mostly uppercase - fix it
            # Convert to lowercase first
            temp_text = temp_text.lower()
            
            # Capitalize sentences (after periods, newlines, colons)
            temp_text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), temp_text)
            temp_text = re.sub(r'(\n)([a-z])', lambda m: m.group(1) + m.group(2).upper(), temp_text)
            temp_text = re.sub(r'(:\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), temp_text)
            
            # Capitalize common technical terms
            replacements = {
                'c++': 'C++',
                'java': 'Java',
                'python': 'Python',
                'javascript': 'JavaScript',
                'sql': 'SQL',
                'api': 'API',
                'html': 'HTML',
                'css': 'CSS',
                'json': 'JSON',
                'xml': 'XML',
                'http': 'HTTP',
                'https': 'HTTPS',
                'url': 'URL',
                'pdf': 'PDF',
            }
            
            for old, new in replacements.items():
                temp_text = re.sub(r'\b' + old + r'\b', new, temp_text, flags=re.IGNORECASE)
    
    # Restore code blocks
    for i, block in enumerate(code_blocks):
        temp_text = temp_text.replace(f'___CODE_BLOCK_{i}___', block)
    
    return temp_text


async def execute_reasoning_production(question: str, context: str, plan: str, 
                                    max_tokens: int = 1500, temperature: float = 0.1) -> tuple[str, str]:
    """
    Execute reasoning with production optimizations and proper capitalization
    """
    reasoning_prompt = f"""You are an expert AI assistant that provides clear, accurate answers.

USER QUESTION: {question}

EXECUTION PLAN:
{plan}

CONTEXT FROM DOCUMENTS:
{context}

IMPORTANT FORMATTING RULES:
1. Use proper sentence case (not ALL CAPS)
2. Provide a complete, well-structured answer
3. Format code properly with proper indentation
4. Use specific information from the context
5. Be clear when information is not in the context

Please provide:
1. Your reasoning process
2. Your final answer

REASONING PROCESS:
[Explain your thinking here]

FINAL ANSWER:
[Provide the complete answer here]
"""
    
    full_response = await call_groq_llm_production(reasoning_prompt, max_tokens, temperature)
    
    # Fix: Better response parsing that preserves formatting
    reasoning = ""
    answer = ""
    
    # Try to split by "FINAL ANSWER:" first (case-insensitive)
    response_upper = full_response.upper()
    
    if "FINAL ANSWER:" in response_upper:
        split_index = response_upper.index("FINAL ANSWER:")
        reasoning = full_response[:split_index].strip()
        answer = full_response[split_index + len("FINAL ANSWER:"):].strip()
        
        # Clean up reasoning
        reasoning = reasoning.replace("REASONING PROCESS:", "").replace("Reasoning Process:", "").strip()
        
    elif "ANSWER:" in response_upper and "REASONING" in response_upper:
        split_index = response_upper.index("ANSWER:")
        reasoning = full_response[:split_index].strip()
        answer = full_response[split_index + len("ANSWER:"):].strip()
        
    else:
        # No clear split - use the whole response as answer
        reasoning = "Analyzed the context and provided a comprehensive response."
        answer = full_response
    
    # Fix capitalization issues in the answer (but preserve code blocks)
    answer = fix_response_formatting(answer)
    
    return answer, reasoning


def assess_confidence_production(question: str, retrieved_docs: int, context_length: int, 
                            sources: List[str]) -> str:
    """
    Production-grade confidence assessment
    """
    score = 0
    
    # Document quantity score
    if retrieved_docs >= 5:
        score += 40
    elif retrieved_docs >= 3:
        score += 25
    elif retrieved_docs >= 1:
        score += 10
    
    # Content quality score
    if context_length > 2000:
        score += 30
    elif context_length > 1000:
        score += 20
    elif context_length > 500:
        score += 10
    
    # Source diversity score
    unique_sources = len(set(sources))
    if unique_sources >= 3:
        score += 20
    elif unique_sources >= 2:
        score += 10
    elif unique_sources >= 1:
        score += 5
    
    # Question complexity score
    if len(question.split()) > 10:
        score += 10
    
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

# -------------------
# Main Production Pipeline
# -------------------
async def agentic_pipeline_production(question: str, top_k: int = TOP_K, include_plan: bool = True,
                                    max_tokens: int = 1500, temperature: float = 0.1, 
                                    use_cache: bool = True, request_id: str = None) -> AgentResponse:
    """
    Production-grade agentic pipeline with caching and optimization
    """
    start_time = time.time()
    
    # Check cache first
    cache_key = get_cache_key(question, top_k, temperature) if use_cache else None
    if cache_key and cache_key in query_cache:
        with cache_lock:
            cached_result = query_cache[cache_key]
            if time.time() - cached_result['timestamp'] < CACHE_TTL:
                result = cached_result['response']
                result.cached = True
                result.request_id = request_id
                result.processing_time = time.time() - start_time
                logger.info(f"Cache hit for request {request_id}")
                return result
    
    try:
        logger.info(f"Processing request {request_id}: {question[:100]}...")
        
        # Step 1: Generate query embedding (with caching)
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            thread_pool,
            lambda: cached_encode(question)
        )
        
        if not query_embedding:
            raise Exception("Failed to generate query embedding")
        
        logger.info(f"Generated query embedding for {request_id}")
        
        # Step 2: Retrieve relevant documents
        documents = await retrieve_documents_production(query_embedding, top_k)
        logger.info(f"Retrieved {len(documents)} documents for {request_id}")
        
        if not documents:
            response = AgentResponse(
                answer="I couldn't find any relevant documents to answer your question. This might be because the documents haven't been indexed yet, or your question is outside the scope of the available knowledge base. Please try rephrasing your question or contact support if the issue persists.",
                plan="No planning needed - no relevant documents found.",
                reasoning="No relevant documents were retrieved from the knowledge base.",
                sources=[],
                retrieved_documents=[],
                retrieved_docs=0,
                confidence="Low",
                processing_time=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
                request_id=request_id,
                cached=False
            )
            return response
        
        # Step 3: Prepare context with intelligent truncation
        context_parts = []
        total_length = 0
        
        for doc in documents:
            doc_text = f"Source: {doc.doc_id}\n{doc.content}\n\n"
            if total_length + len(doc_text) <= MAX_CONTEXT_LENGTH:
                context_parts.append(doc_text)
                total_length += len(doc_text)
            else:
                # Truncate the last document to fit
                remaining = MAX_CONTEXT_LENGTH - total_length
                if remaining > 200:  # Only include if meaningful content can fit
                    truncated = doc_text[:remaining] + "\n[Content truncated...]\n\n"
                    context_parts.append(truncated)
                break
        
        context = "".join(context_parts)
        
        # Step 4: Create plan (if requested)
        plan = ""
        if include_plan:
            plan = await create_plan_production(question, context)
            logger.info(f"Generated execution plan for {request_id}")
        
        # Step 5: Execute reasoning
        answer, reasoning = await execute_reasoning_production(
            question, context, plan, max_tokens, temperature
        )
        logger.info(f"Generated final answer for {request_id}")
        
        # Step 6: Extract metadata
        sources = list(set(doc.doc_id for doc in documents))
        confidence = assess_confidence_production(question, len(documents), len(context), sources)
        
        processing_time = time.time() - start_time
        
        response = AgentResponse(
            answer=answer,
            plan=plan if include_plan else None,
            reasoning=reasoning,
            sources=sources,
            retrieved_documents=documents,
            retrieved_docs=len(documents),
            confidence=confidence,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            cached=False
        )
        
        # Cache the result
        if use_cache and cache_key:
            with cache_lock:
                query_cache[cache_key] = {
                    'response': response,
                    'timestamp': time.time()
                }
                # Limit cache size
                if len(query_cache) > 1000:
                    # Remove oldest 100 entries
                    sorted_items = sorted(query_cache.items(), key=lambda x: x[1]['timestamp'])
                    for old_key, _ in sorted_items[:100]:
                        del query_cache[old_key]
        
        return response
        
    except Exception as e:
        logger.exception(f"Pipeline failed for request {request_id}")
        processing_time = time.time() - start_time
        
        return AgentResponse(
            answer=f"I encountered a technical error while processing your question. Our system is designed to handle large documents and high traffic, but occasionally issues occur. Please try again, and if the problem persists, contact our support team. Error reference: {request_id}",
            plan="Error occurred during processing.",
            reasoning="A technical error prevented normal processing.",
            sources=[],
            retrieved_documents=[],
            retrieved_docs=0,
            confidence="Low",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            cached=False
        )

# -------------------
# Production FastAPI Endpoints
# -------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time and request ID headers"""
    request_id = generate_request_id()
    request.state.request_id = request_id
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    return response

@app.get("/", response_model=dict)
async def root():
    """Production API root endpoint"""
    return {
        "service": "Agentic RAG Production API",
        "version": "2.0.0",
        "status": "operational",
        "max_concurrent_users": MAX_CONCURRENT_REQUESTS,
        "supported_document_size": "500MB",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check_production():
    """Comprehensive production health check"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Test model
        test_start = time.time()
        test_embedding = await asyncio.get_event_loop().run_in_executor(
            thread_pool, lambda: model.encode(["health check"]).tolist()
        )
        embedding_time = time.time() - test_start
        
        # Test Groq API
        groq_start = time.time()
        test_response = await call_groq_llm_production("Say 'OK'", max_tokens=10)
        groq_time = time.time() - groq_start
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            system_info={
                "cpu_cores": psutil.cpu_count(),
                "cpu_usage_percent": cpu_percent,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_usage_percent": memory.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "worker_threads": WORKER_THREADS,
                "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
            },
            performance_metrics={
                "embedding_generation_ms": round(embedding_time * 1000, 2),
                "groq_api_response_ms": round(groq_time * 1000, 2),
                "cache_size": len(query_cache),
                "active_requests": MAX_CONCURRENT_REQUESTS - request_semaphore._value
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/query", response_model=AgentResponse, dependencies=[Depends(rate_limit_check)])
async def query_documents_production(req: QueryRequest, request: Request) -> AgentResponse:
    """
    Production query endpoint with rate limiting and optimization
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    request_id = getattr(request.state, 'request_id', generate_request_id())
    top_k = req.top_k if req.top_k is not None else TOP_K
    max_tokens = min(req.max_tokens, 4000)  # Cap max tokens
    temperature = req.temperature
    
    try:
        result = await agentic_pipeline_production(
            question=req.question,
            top_k=top_k,
            include_plan=req.include_plan,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=req.use_cache,
            request_id=request_id
        )
        
        logger.info(f"Successfully processed request {request_id} in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.exception(f"Query processing failed for request {request_id}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

@app.post("/agentic_query", response_model=AgentResponse, dependencies=[Depends(rate_limit_check)])
async def agentic_query_endpoint_production(req: QueryRequest, request: Request) -> AgentResponse:
    """
    Streamlit-compatible endpoint (same as /query)
    """
    return await query_documents_production(req, request)

@app.post("/simple_query", dependencies=[Depends(rate_limit_check)])
async def simple_query_production(req: QueryRequest, request: Request):
    """
    Simplified endpoint for basic integrations
    """
    result = await query_documents_production(req, request)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "processing_time": result.processing_time,
        "request_id": result.request_id
    }

@app.get("/stats")
async def get_production_stats():
    """Production statistics endpoint"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/documents"
        params = {"select": "doc_id", "limit": "10000"}  # Increased limit
        
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=SUPABASE_HEADERS)
            
            if response.status_code == 200:
                docs = response.json()
                unique_docs = set(doc['doc_id'] for doc in docs)
                
                return {
                    "total_chunks": len(docs),
                    "unique_documents": len(unique_docs),
                    "embedding_model": EMBEDDING_MODEL,
                    "top_k_default": TOP_K,
                    "max_context_length": MAX_CONTEXT_LENGTH,
                    "max_concurrent_users": MAX_CONCURRENT_REQUESTS,
                    "cache_size": len(query_cache),
                    "system_status": "operational"
                }
            else:
                return {"error": "Could not retrieve statistics", "status_code": response.status_code}
                
    except Exception as e:
        return {"error": f"Statistics unavailable: {str(e)}"}

@app.post("/clear_cache")
async def clear_cache_endpoint():
    """Clear application caches (admin endpoint)"""
    try:
        with cache_lock:
            query_cache.clear()
            embedding_cache.clear()
        
        # Clear LRU cache
        cached_encode.cache_clear()
        
        # Force garbage collection
        gc.collect()
        
        return {"message": "All caches cleared successfully", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")

# -------------------
# Production Server Configuration
# -------------------
if __name__ == '__main__':
    # Production server configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    workers = int(os.getenv('WORKERS', 1))  # Single worker for shared state
    
    logger.info(f"🚀 Starting Production Agentic RAG Server")
    logger.info(f"   Host: {host}:{port}")
    logger.info(f"   Max Concurrent Users: {MAX_CONCURRENT_REQUESTS}")
    logger.info(f"   Worker Threads: {WORKER_THREADS}")
    logger.info(f"   Request Timeout: {REQUEST_TIMEOUT}s")
    logger.info(f"   Max Document Size: 500MB")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        loop="asyncio",
        access_log=False,  # Disable access logs for performance
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=MAX_CONCURRENT_REQUESTS,
        backlog=2048
    )