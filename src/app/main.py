import sys
import os
import argparse
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_core.rag_chain import RAGChain
from rag_core.retriever import Retriever
from rag_core.generator import Generator

@dataclass
class ChunkMetadata:
    """Lưu trữ metadata cho mỗi chunk"""
    chunk_id: int
    content: str
    heading: Optional[str]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatBot:
    def __init__(
        self,
        embeddings_dir: Optional[str] = None,
        top_k: int = 3,
        model_filename: Optional[str] = None,
        n_ctx: int = 2048,
        verbose: bool = False,
    ):
        logger.info("=" * 60)
        logger.info("Initializing ChatBot with RAG System")
        logger.info("=" * 60)
        
        try:
            logger.info("Initializing RAG Chain...")
            self.rag_chain = RAGChain(
                embeddings_dir=embeddings_dir,
                top_k=top_k,
                model_filename=model_filename,
                n_ctx=n_ctx,
                verbose=verbose,
            )
            
            logger.info("ChatBot initialized successfully!")
            
            self.conversation_history = []
            self.start_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatBot: {str(e)}")
            raise

    def process_query(self, query: str, stream: bool = False):
        try:
            logger.info(f"\nProcessing query: {query}")
            
            response = self.rag_chain.generate(query, stream=stream, max_tokens=512)
            
            self.conversation_history.append({
                "type": "user",
                "content": query,
                "timestamp": datetime.now()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def display_response(self, response: str):
        print("\n" + "=" * 60)
        print("ASSISTANT:")
        print("=" * 60)
        print(response)

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
            if len(content) > 300:
                content = content[:300] + "...[truncated]"
            
            print(content)

    def clear_history(self):
        self.conversation_history = []
        print("\n🗑️  Conversation history cleared.")

    def run(self):
        print("\n" + "╔" + "=" * 58 + "╗")
        print("║" + " " * 15 + "🤖 RAG CHATBOT - Interactive Mode" + " " * 10 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\nWelcome to the RAG-based document Q&A chatbot!")
        print("Type 'help' for commands or just ask a question.")
        print("Type 'quit' or 'exit' to close the application.\n")
        
        try:
            while True:
                print("-" * 60)
                try:
                    user_input = input("\n📝 You: ").strip()
                except KeyboardInterrupt:
                    print("\n\nChatbot interrupted by user.")
                    break
                
                if not user_input:
                    print("Please enter a question or command.")
                    continue
                
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
                
                print("\n⏳ Processing your question...")
                print("-" * 60)
                print("📝 Response (streaming):")
                print("-" * 60)
                
                response_stream = self.rag_chain.generate(user_input, stream=True, max_tokens=512)
                
                full_response = ""
                for token in response_stream:
                    print(token, end="", flush=True)
                    full_response += token
                
                print("\n")
                
                self.conversation_history.append({
                    "type": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now()
                })
        
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")
            print(f"\n❌ An error occurred: {str(e)}")
        
        finally:
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
    parser = argparse.ArgumentParser(
        description="RAG-based Document Q&A Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run interactive mode
  python main.py --verbose               # Run with verbose logging
  python main.py --top-k 5               # Retrieve top 5 documents
  python main.py --model-file "*.gguf"   # Specify custom model file
        """
    )
    
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Path to embeddings directory (default: auto-detect)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve (default: 3)"
    )
    
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="GGUF model filename pattern (default: *Q4_K_M.gguf)"
    )
    
    parser.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context window size (default: 2048)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        print("\n" + "═" * 60)
        print("  RAG-based Document Question Answering System")
        print("═" * 60 + "\n")
        
        chatbot = ChatBot(
            embeddings_dir=args.embeddings_dir,
            top_k=args.top_k,
            model_filename=args.model_file,
            n_ctx=args.context_size,
            verbose=args.verbose
        )
        chatbot.run()
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
