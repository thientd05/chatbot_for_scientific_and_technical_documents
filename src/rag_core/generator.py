from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.schema import BaseGenerator # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI research assistant. You provide accurate, scientific responses based on provided context, admitting uncertainty when appropriate. When asked about information beyond the provided context, you should acknowledge the limitation and stick to what's supported by the context."""

class RAGGenerator(BaseGenerator):
    """Advanced generator using Mixtral-8x7B with optimized prompting for scientific text"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        load_in_4bit: bool = True
    ):
        """Initialize the generator with Mixtral model"""
        super().__init__()
        
        logger.info(f"Initializing RAGGenerator with model: {model_name}")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model in 4-bit precision for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            torch_dtype=torch.float16
        )
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        
        # Define the main prompt template
        self.prompt_template = PromptTemplate(
            template="""<s>[INST] {system_prompt}

Context information is below:
{context}

Given the context above, please respond to the following: {query} [/INST]""",
            input_variables=["context", "query", "system_prompt"]
        )

    def _format_context(self, context_docs: List[Dict]) -> str:
        """Format the context documents into a clear structure"""
        formatted_context = []
        
        for i, doc in enumerate(context_docs, 1):
            # Extract and format metadata
            metadata = doc.get("metadata", {})
            section = metadata.get("section", "Unknown Section")
            relevance = metadata.get("relevance_score", 0)
            
            # Format the context piece
            context_piece = f"""[{i}] From section '{section}' (relevance: {relevance:.2f}):
{doc.get('content', 'No content available')}
"""
            formatted_context.append(context_piece)
            
        return "\n\n".join(formatted_context)

    def generate(
        self,
        query: str,
        context_docs: List[Dict],
        **kwargs
    ) -> str:
        """Generate a response based on the query and context"""
        try:
            logger.info(f"Generating response for query: {query}")
            
            # Format context
            formatted_context = self._format_context(context_docs)
            
            # Create the full prompt
            full_prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                context=formatted_context,
                query=query
            )
            
            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3000  # Leave room for generation
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean up response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(full_prompt, "").strip()
            
            logger.info("Successfully generated response")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def stream(
        self,
        query: str,
        context_docs: List[Dict],
        **kwargs
    ):
        """Stream the generated response token by token"""
        try:
            formatted_context = self._format_context(context_docs)
            full_prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                context=formatted_context,
                query=query
            )
            
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3000
            ).to(self.device)
            
            # Stream generation
            streamed_tokens = []
            for outputs in self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                streaming=True
            ):
                next_token = outputs[0][-1:]
                next_token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
                streamed_tokens.append(next_token_text)
                yield next_token_text
                
        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}")
            raise
