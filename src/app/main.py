"""
Main chatbot application implementing an interactive chat loop
using the RAG (Retrieval-Augmented Generation) system.
"""

import sys
import os
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_core.rag_chain import RAGChain, RAGResponse
from rag_core.retriever import EnhancedRAGRetriever
from rag_core.generator import RAGGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatBot:
    """Interactive chatbot with RAG capabilities"""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        generator_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        retriever_top_k: int = 5,
        enable_reranking: bool = True,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        collection_name: str = "scientific_papers"
    ):
        """
        Initialize the chatbot with RAG chain components.
        
        Args:
            embedding_model: Name of the embedding model from HuggingFace
            generator_model: Name of the generator model
            retriever_top_k: Number of top documents to retrieve
            enable_reranking: Whether to enable cross-encoder reranking
            milvus_host: Host for Milvus vector database
            milvus_port: Port for Milvus vector database
            collection_name: Name of the Milvus collection
        """
        logger.info("=" * 60)
        logger.info("Initializing ChatBot with RAG System")
        logger.info("=" * 60)
        
        try:
            # Initialize Milvus connection arguments
            milvus_connection_args = {
                "host": milvus_host,
                "port": milvus_port,
                "collection_name": collection_name
            }
            
            # Initialize retriever
            logger.info(f"Initializing retriever with model: {embedding_model}")
            self.retriever = EnhancedRAGRetriever(
                embedding_model_name=embedding_model,
                milvus_connection_args=milvus_connection_args,
                top_k=retriever_top_k,
                score_threshold=0.6,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # Initialize generator
            logger.info(f"Initializing generator with model: {generator_model}")
            self.generator = RAGGenerator(
                model_name=generator_model,
                device="cuda",
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                load_in_4bit=True
            )
            
            # Initialize RAG chain
            logger.info("Initializing RAG chain with reranking")
            self.rag_chain = RAGChain(
                retriever=self.retriever,
                generator=self.generator,
                enable_reranking=enable_reranking,
                rerank_top_k=100,
                device="cuda"
            )
            
            logger.info("ChatBot initialized successfully!")
            
            # Conversation history
            self.conversation_history = []
            self.start_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatBot: {str(e)}")
            raise

    def process_query(self, query: str) -> tuple[str, Optional[RAGResponse]]:
        """
        Process a user query and return the response.
        
        Args:
            query: The user's question
            
        Returns:
            Tuple of (response_text, rag_response_object)
        """
        try:
            logger.info(f"\nProcessing query: {query}")
            
            # Generate response using RAG chain
            rag_response = self.rag_chain.generate(query)
            
            # Add to conversation history
            self.conversation_history.append({
                "type": "user",
                "content": query,
                "timestamp": datetime.now()
            })
            
            self.conversation_history.append({
                "type": "assistant",
                "content": rag_response.answer,
                "timestamp": datetime.now(),
                "num_sources": len(rag_response.source_documents)
            })
            
            return rag_response.answer, rag_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}", None

    def display_response(self, response: str, rag_response: Optional[RAGResponse] = None):
        """Display the response with formatting"""
        print("\n" + "=" * 60)
        print("ASSISTANT:")
        print("=" * 60)
        print(response)
        
        if rag_response and rag_response.source_documents:
            print("\n" + "-" * 60)
            print(f"📚 Sources ({len(rag_response.source_documents)} documents used):")
            print("-" * 60)
            for i, doc in enumerate(rag_response.source_documents[:3], 1):  # Show top 3 sources
                metadata = doc.metadata
                section = metadata.get("section", "Unknown")
                relevance = metadata.get("relevance_score", metadata.get("score", 0))
                print(f"\n[{i}] {section}")
                if "page" in metadata:
                    print(f"    Page: {metadata['page']}")
                print(f"    Relevance Score: {relevance:.3f}")
                print(f"    Preview: {doc.page_content[:150]}...")

    def display_help(self):
        """Display help message"""
        help_text = """
╔════════════════════════════════════════════════════════════╗
║       RAG CHATBOT - Interactive Help                       ║
╚════════════════════════════════════════════════════════════╝

Commands:
  help    - Show this help message
  history - Show conversation history
  clear   - Clear conversation history
  quit    - Exit the chatbot
  exit    - Exit the chatbot

Examples of queries:
  • "What is attention mechanism?"
  • "Explain transformer architecture"
  • "How does self-attention work?"
  • "Tell me about positional encoding"

Tips:
  - Ask specific questions for better results
  - The chatbot uses semantic search to find relevant documents
  - Responses are based on the provided document collection
  - Type 'quit' or 'exit' to close the application

════════════════════════════════════════════════════════════
        """
        print(help_text)

    def show_conversation_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            print("\n📭 No conversation history yet.")
            return
        
        print("\n" + "=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)
        
        for i, msg in enumerate(self.conversation_history, 1):
            msg_type = "👤 USER" if msg["type"] == "user" else "🤖 ASSISTANT"
            timestamp = msg["timestamp"].strftime("%H:%M:%S")
            
            print(f"\n[{i}] {msg_type} ({timestamp})")
            print("-" * 60)
            
            content = msg["content"]
            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "...[truncated]"
            
            print(content)
            
            if msg["type"] == "assistant" and "num_sources" in msg:
                print(f"(Used {msg['num_sources']} source documents)")

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("\n🗑️  Conversation history cleared.")

    def run(self):
        """Main chat loop"""
        print("\n" + "╔" + "=" * 58 + "╗")
        print("║" + " " * 15 + "🤖 RAG CHATBOT - Interactive Mode" + " " * 10 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\nWelcome to the RAG-based scientific chatbot!")
        print("Type 'help' for commands or just ask a question.")
        print("Type 'quit' or 'exit' to close the application.\n")
        
        try:
            while True:
                # Get user input
                print("-" * 60)
                try:
                    user_input = input("\n📝 You: ").strip()
                except KeyboardInterrupt:
                    print("\n\nChatbot interrupted by user.")
                    break
                
                # Handle empty input
                if not user_input:
                    print("Please enter a question or command.")
                    continue
                
                # Handle special commands
                if user_input.lower() in ["quit", "exit"]:
                    print("\n👋 Thank you for using the RAG Chatbot. Goodbye!")
                    break
                
                elif user_input.lower() == "help":
                    self.display_help()
                    continue
                
                elif user_input.lower() == "history":
                    self.show_conversation_history()
                    continue
                
                elif user_input.lower() == "clear":
                    self.clear_history()
                    continue
                
                # Process regular query
                print("\n⏳ Processing your question...")
                response, rag_response = self.process_query(user_input)
                self.display_response(response, rag_response)
        
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"\n❌ An error occurred: {str(e)}")
        
        finally:
            # Summary
            elapsed_time = datetime.now() - self.start_time
            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            print(f"Session Duration: {elapsed_time}")
            print(f"Total Messages: {len(self.conversation_history)}")
            print(f"Total Exchanges: {len(self.conversation_history) // 2}")
            print("=" * 60)


def main():
    """Main entry point"""
    try:
        # Initialize chatbot
        chatbot = ChatBot(
            embedding_model="BAAI/bge-large-en-v1.5",
            generator_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            retriever_top_k=5,
            enable_reranking=True,
            milvus_host="localhost",
            milvus_port="19530",
            collection_name="scientific_papers"
        )
        
        # Run chat loop
        chatbot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
