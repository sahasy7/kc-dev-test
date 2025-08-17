from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Dict, Optional
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from datetime import datetime
import uuid
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for environment variables"""
    URL = os.getenv('uri')
    KEY = os.getenv('key')
    DATABASE = os.getenv('database')
    CONTAINER_1 = os.getenv('container_1')  # Vector search container
    CONTAINER_2 = os.getenv('container_2')  # Conversation history container
    EMB_MODEL = os.getenv('emb_model')
    EMBED_KEY = os.getenv('embed_key')
    EMBED_URL = os.getenv('embed_url')

class QueryRequest(BaseModel):
    """Request model for chat queries"""
    phone_number: str = Field(..., description="User's phone number", min_length=10)
    query: str = Field(..., description="User's question", min_length=1, max_length=1000)

class QueryResponse(BaseModel):
    """Response model for chat queries"""
    success: bool
    response: str
    phone_number: str
    timestamp: str
    conversation_saved: bool

class VectorSearchResult(BaseModel):
    """Model for vector search results"""
    content: str
    similarity_score: float



class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str

class RAGService:
    """Service class for RAG operations"""
    
    def __init__(self):
        self.config = Config()
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure Cosmos DB and OpenAI clients"""
        try:
            # Cosmos DB client
            self.cosmos_client = CosmosClient(url=self.config.URL, credential=self.config.KEY)
            self.database = self.cosmos_client.get_database_client(self.config.DATABASE)
            self.vector_container = self.database.get_container_client(self.config.CONTAINER_1)
            self.conversation_container = self.database.get_container_client(self.config.CONTAINER_2)
            
            # OpenAI client
            self.openai_client = AzureOpenAI(
                api_key=self.config.EMBED_KEY,
                api_version="2024-12-01-preview",
                azure_endpoint=self.config.EMBED_URL
            )
            
            # Embedding client
            self.embedding_client = AzureOpenAI(
                api_key=self.config.EMBED_KEY,
                api_version="2024-02-01",
                azure_endpoint=self.config.EMBED_URL
            )
            
            logger.info("Clients initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for given text"""
        try:
            response = self.embedding_client.embeddings.create(
                input=[text], 
                model=self.config.EMB_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def vector_search(self, query_text: str, k: int = 3) -> List[VectorSearchResult]:
        """Perform vector search"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings(query_text)
            
            # Search query
            search_query = f"""
            SELECT TOP {k} c.content, VectorDistance(c.embedding, @embedding) AS similarity_score
            FROM c
            ORDER BY VectorDistance(c.embedding, @embedding)
            """
            
            results = self.vector_container.query_items(
                query=search_query,
                parameters=[{"name": "@embedding", "value": query_embedding}],
                enable_cross_partition_query=True
            )

            return [VectorSearchResult(**result) for result in results]
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise

    async def save_conversation_pair(self, phone_number: str, query: str, response: str) -> bool:
        """Save query and assistant response"""
        try:
            current_time = datetime.utcnow().isoformat()
            
            query_doc = {
                "id": str(uuid.uuid4()),
                "phone_number": phone_number,
                "message_type": "query",
                "content": query,
                "timestamp": current_time
            }
            response_doc = {
                "id": str(uuid.uuid4()),
                "phone_number": phone_number,
                "message_type": "response",
                "content": response,
                "timestamp": current_time
            }

            self.conversation_container.create_item(query_doc)
            self.conversation_container.create_item(response_doc)
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    async def get_context_for_openai(self, phone_number: str) -> List[Dict]:
        """Fetch last 2 conversations (query + response)"""
        try:
            query = """
                SELECT TOP 4 c.message_type, c.content, c.timestamp 
                FROM c 
                WHERE c.phone_number = @phone_number 
                ORDER BY c.timestamp DESC
            """
            parameters = [{"name": "@phone_number", "value": phone_number}]

            messages = list(self.conversation_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=phone_number
            ))

            messages.reverse()  # chronological order
            openai_messages = []
            for msg in messages:
                role = "user" if msg["message_type"] == "query" else "assistant"
                openai_messages.append({"role": role, "content": msg["content"]})
            return openai_messages

        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return []

    async def process_query(self, phone_number: str, query: str, k: int = 3) -> tuple[str, bool]:
        """Process the complete RAG query"""
        try:
            # Get conversation context
            context_chat = await self.get_context_for_openai(phone_number)
            
            # Perform vector search
            vector_results = self.vector_search(query_text=query, k=k)
            context_doc = "\n".join([doc.content for doc in vector_results])
            
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": f"You are a helpful assistant for user {phone_number}. Answer questions based on the provided documents and conversation history. If you don't know something, say so."}
            ]
            messages.extend(context_chat)
            messages.append({
                "role": "user", 
                "content": f"Documents: {context_doc}\n\nCurrent Question: {query}"
            })

            # Get response from OpenAI
            response = self.openai_client.chat.completions.create(
                messages=messages,
                max_completion_tokens=None,
                temperature=1.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                model="gpt-4.1-mini"
            )
            
            result = response.choices[0].message.content
            
            # Save conversation
            conversation_saved = await self.save_conversation_pair(phone_number, query, result)
            
            return result, conversation_saved
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="A Retrieval-Augmented Generation API for conversational AI",
    version="1.0.0"
)

# Initialize RAG service
rag_service = RAGService()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Main chat endpoint for processing user queries"""
    try:
        logger.info(f"Processing query from {request.phone_number}: {request.query[:50]}...")
        
        # Process the query
        response_text, conversation_saved = await rag_service.process_query(
            phone_number=request.phone_number,
            query=request.query
        )
        
        return QueryResponse(
            success=True,
            response=response_text,
            phone_number=request.phone_number,
            timestamp=datetime.utcnow().isoformat(),
            conversation_saved=conversation_saved
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


