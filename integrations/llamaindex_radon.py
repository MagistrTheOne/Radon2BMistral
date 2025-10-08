"""
LlamaIndex Integration for RADON
Vector database and retrieval-augmented generation with RADON
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field

from llama_index.llms.base import LLM, ChatMessage, MessageRole
from llama_index.llms.llm import LLMMetadata
from llama_index.llms.custom import CustomLLM
from llama_index.embeddings.base import BaseEmbedding
from llama_index.embeddings.custom import CustomEmbedding
from llama_index.core.llms import ChatResponse, CompletionResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import numpy as np


class RADONLLM(CustomLLM):
    """
    LlamaIndex LLM wrapper for RADON
    """
    
    model_name: str = "MagistrTheOne/RadonSAI-Pretrained"
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    device: str = "auto"
    torch_dtype: str = "float16"
    
    # Internal attributes
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load RADON model and tokenizer"""
        try:
            print(f"Loading RADON model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Determine torch dtype
            torch_dtype = torch.float16 if self.torch_dtype == "float16" else torch.float32
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            print(f"✅ RADON model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load RADON model: {e}")
            raise
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata"""
        return LLMMetadata(
            context_window=8192,
            num_output=self.max_tokens,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="radon"
        )
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete text generation"""
        
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model not loaded")
        
        # Override parameters with kwargs
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        repetition_penalty = kwargs.get("repetition_penalty", self.repetition_penalty)
        
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )
            
            # Decode response
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return CompletionResponse(text=response)
            
        except Exception as e:
            return CompletionResponse(text=f"Error: {str(e)}")
    
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat completion"""
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Generate response
        completion = self.complete(prompt, **kwargs)
        
        # Create chat response
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=completion.text
            )
        )
    
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a single prompt"""
        formatted_prompt = ""
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == MessageRole.SYSTEM:
                formatted_prompt += f"System: {content}\n\n"
            elif role == MessageRole.USER:
                formatted_prompt += f"User: {content}\n\n"
            elif role == MessageRole.ASSISTANT:
                formatted_prompt += f"Assistant: {content}\n\n"
        
        formatted_prompt += "Assistant:"
        return formatted_prompt
    
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Stream text completion"""
        
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model not loaded")
        
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            generated_tokens = []
            
            with torch.no_grad():
                for _ in range(max_tokens):
                    # Forward pass
                    outputs = self._model(**inputs)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Sample next token
                    if temperature > 0:
                        probs = torch.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for EOS
                    if next_token.item() == self._tokenizer.eos_token_id:
                        break
                    
                    # Add to generated tokens
                    generated_tokens.append(next_token.item())
                    
                    # Update inputs for next iteration
                    inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
                    
                    # Yield partial result
                    partial_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    yield CompletionResponse(text=partial_text)
            
        except Exception as e:
            yield CompletionResponse(text=f"Error: {str(e)}")
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        """Stream chat completion"""
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Stream response
        for completion in self.stream_complete(prompt, **kwargs):
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=completion.text
                )
            )


class RADONEmbedding(CustomEmbedding):
    """
    RADON-based embedding model for LlamaIndex
    """
    
    model_name: str = "MagistrTheOne/RadonSAI-Pretrained"
    device: str = "auto"
    torch_dtype: str = "float16"
    max_length: int = 512
    
    # Internal attributes
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load RADON model for embeddings"""
        try:
            print(f"Loading RADON embedding model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Determine torch dtype
            torch_dtype = torch.float16 if self.torch_dtype == "float16" else torch.float32
            
            # Load model for embeddings (use base model)
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            print(f"✅ RADON embedding model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load RADON embedding model: {e}")
            raise
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding"""
        return self._get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding"""
        
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model not loaded")
        
        try:
            # Tokenize text
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                # Convert to CPU and normalize
                embeddings = embeddings.cpu().numpy()
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings[0].tolist()
            
        except Exception as e:
            print(f"Embedding error: {e}")
            # Return zero embedding as fallback
            return [0.0] * 2048  # Default embedding size
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings"""
        return [self._get_text_embedding(text) for text in texts]


class RADONRAG:
    """
    Retrieval-Augmented Generation with RADON
    """
    
    def __init__(
        self,
        llm: RADONLLM,
        embedding: RADONEmbedding,
        documents: Optional[List[str]] = None
    ):
        self.llm = llm
        self.embedding = embedding
        self.documents = documents or []
        self.document_embeddings = []
        
        if self.documents:
            self._index_documents()
    
    def _index_documents(self):
        """Index documents for retrieval"""
        print("Indexing documents...")
        
        for doc in self.documents:
            embedding = self.embedding._get_text_embedding(doc)
            self.document_embeddings.append(embedding)
        
        print(f"✅ Indexed {len(self.documents)} documents")
    
    def add_documents(self, documents: List[str]):
        """Add new documents to the index"""
        for doc in documents:
            embedding = self.embedding._get_text_embedding(doc)
            self.document_embeddings.append(embedding)
            self.documents.append(doc)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant documents"""
        
        if not self.documents:
            return []
        
        # Get query embedding
        query_embedding = self.embedding._get_query_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc_embedding in self.document_embeddings:
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
    def generate(
        self, 
        query: str, 
        context_documents: Optional[List[str]] = None,
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> str:
        """Generate response with retrieved context"""
        
        # Retrieve relevant documents if not provided
        if context_documents is None:
            context_documents = self.retrieve(query, top_k=3)
        
        # Format context
        context = "\n\n".join(context_documents) if context_documents else ""
        
        # Create prompt with context
        if context:
            prompt = f"""Контекст:
{context}

Вопрос: {query}

Ответ на основе контекста:"""
        else:
            prompt = f"Вопрос: {query}\n\nОтвет:"
        
        # Generate response
        response = self.llm.complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.text
    
    def chat(
        self,
        messages: List[ChatMessage],
        context_documents: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResponse:
        """Chat with retrieved context"""
        
        # Get the last user message
        user_message = None
        for message in reversed(messages):
            if message.role == MessageRole.USER:
                user_message = message.content
                break
        
        if user_message is None:
            return self.llm.chat(messages, **kwargs)
        
        # Retrieve relevant documents
        if context_documents is None:
            context_documents = self.retrieve(user_message, top_k=3)
        
        # Format context
        context = "\n\n".join(context_documents) if context_documents else ""
        
        # Create enhanced messages with context
        enhanced_messages = []
        if context:
            enhanced_messages.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Используй следующий контекст для ответа:\n\n{context}"
                )
            )
        
        enhanced_messages.extend(messages)
        
        # Generate response
        return self.llm.chat(enhanced_messages, **kwargs)


# Factory functions
def create_radon_llm(**kwargs) -> RADONLLM:
    """Create RADON LLM instance"""
    return RADONLLM(**kwargs)


def create_radon_embedding(**kwargs) -> RADONEmbedding:
    """Create RADON embedding instance"""
    return RADONEmbedding(**kwargs)


def create_radon_rag(
    documents: Optional[List[str]] = None,
    llm_kwargs: Optional[Dict] = None,
    embedding_kwargs: Optional[Dict] = None
) -> RADONRAG:
    """Create RADON RAG instance"""
    
    llm_kwargs = llm_kwargs or {}
    embedding_kwargs = embedding_kwargs or {}
    
    llm = create_radon_llm(**llm_kwargs)
    embedding = create_radon_embedding(**embedding_kwargs)
    
    return RADONRAG(llm=llm, embedding=embedding, documents=documents)


# Example usage
def example_usage():
    """Example usage of RADON with LlamaIndex"""
    
    # Create RADON LLM
    radon_llm = create_radon_llm(
        max_tokens=100,
        temperature=0.7
    )
    
    # Basic completion
    prompt = "Машинное обучение - это"
    response = radon_llm.complete(prompt)
    print(f"Prompt: {prompt}")
    print(f"RADON Response: {response.text}")
    
    # Chat completion
    messages = [
        ChatMessage(role=MessageRole.USER, content="Объясни что такое нейронные сети")
    ]
    
    chat_response = radon_llm.chat(messages)
    print(f"Chat Response: {chat_response.message.content}")
    
    # RAG example
    documents = [
        "Машинное обучение - это подраздел искусственного интеллекта",
        "Нейронные сети имитируют работу человеческого мозга",
        "Глубокое обучение использует многослойные нейронные сети"
    ]
    
    rag = create_radon_rag(documents=documents)
    
    query = "Что такое машинное обучение?"
    rag_response = rag.generate(query)
    print(f"RAG Response: {rag_response}")
    
    # Chat with RAG
    chat_messages = [
        ChatMessage(role=MessageRole.USER, content="Как работают нейронные сети?")
    ]
    
    rag_chat_response = rag.chat(chat_messages)
    print(f"RAG Chat Response: {rag_chat_response.message.content}")


if __name__ == "__main__":
    example_usage()
