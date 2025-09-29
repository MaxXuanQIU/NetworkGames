"""
LLM interface abstraction layer
Unified calling interface supporting multiple LLM models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import logging
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOCK = "mock"  # For testing


class LLMResponse:
    """LLM response wrapper class"""
    
    def __init__(self, content: str, model: str, usage: Optional[Dict] = None, 
                 response_time: float = 0.0, error: Optional[str] = None):
        self.content = content.strip()
        self.model = model
        self.usage = usage or {}
        self.response_time = response_time
        self.error = error
        self.success = error is None
    
    def is_cooperate(self) -> bool:
        """Determine if the response is cooperation"""
        return self.content.upper() in ["COOPERATE", "COOPERATION", "合作"]
    
    def is_defect(self) -> bool:
        """Determine if the response is defection"""
        return self.content.upper() in ["DEFECT", "DEFECTION", "背叛"]
    
    def is_valid_action(self) -> bool:
        """Determine if the response is a valid game action"""
        return self.is_cooperate() or self.is_defect()


class BaseLLMInterface(ABC):
    """Base class for LLM interfaces"""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response"""
        pass
    
    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """Get provider type"""
        pass
    
    def validate_response(self, response: LLMResponse) -> bool:
        """Validate if the response is valid"""
        if not response.success:
            self.logger.error(f"LLM response error: {response.error}")
            return False
        
        if not response.is_valid_action():
            self.logger.warning(f"Invalid action response: {response.content}")
            return False
        
        return True


class OpenAIInterface(BaseLLMInterface):
    """OpenAI interface implementation"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate OpenAI response"""
        import time
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,  # Limit output length
                temperature=kwargs.get("temperature", 0.7),
                **self.kwargs
            )
            
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content="",
                model=self.model_name,
                response_time=response_time,
                error=str(e)
            )
    
    def get_provider(self) -> LLMProvider:
        return LLMProvider.OPENAI


class AnthropicInterface(BaseLLMInterface):
    """Anthropic interface implementation"""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: str = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate Anthropic response"""
        import time
        start_time = time.time()
        
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                **self.kwargs
            )
            
            content = response.content[0].text
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content="",
                model=self.model_name,
                response_time=response_time,
                error=str(e)
            )
    
    def get_provider(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC


class GoogleInterface(BaseLLMInterface):
    """Google interface implementation"""
    
    def __init__(self, model_name: str = "gemini-pro", api_key: str = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate Google response"""
        import time
        start_time = time.time()
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": 50,
                    "temperature": kwargs.get("temperature", 0.7),
                    **self.kwargs
                }
            )
            
            content = response.text
            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split())
            }
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content="",
                model=self.model_name,
                response_time=response_time,
                error=str(e)
            )
    
    def get_provider(self) -> LLMProvider:
        return LLMProvider.GOOGLE


class MockLLMInterface(BaseLLMInterface):
    """Mock LLM interface for testing"""
    
    def __init__(self, model_name: str = "mock-model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.cooperation_rate = kwargs.get("cooperation_rate", 0.5)
        self.response_delay = kwargs.get("response_delay", 0.1)
    
    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response"""
        import time
        import random
        
        start_time = time.time()
        
        # Simulate response delay
        await asyncio.sleep(self.response_delay)
        
        # Randomly generate response based on cooperation rate
        if random.random() < self.cooperation_rate:
            content = "COOPERATE"
        else:
            content = "DEFECT"
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=content,
            model=self.model_name,
            response_time=response_time
        )
    
    def get_provider(self) -> LLMProvider:
        return LLMProvider.MOCK


class LLMFactory:
    """LLM factory class"""
    
    @staticmethod
    def create_llm(provider: LLMProvider, model_name: str, api_key: str = None, **kwargs) -> BaseLLMInterface:
        """Create LLM instance"""
        if provider == LLMProvider.OPENAI:
            return OpenAIInterface(model_name, api_key, **kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicInterface(model_name, api_key, **kwargs)
        elif provider == LLMProvider.GOOGLE:
            return GoogleInterface(model_name, api_key, **kwargs)
        elif provider == LLMProvider.MOCK:
            return MockLLMInterface(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseLLMInterface:
        """Create LLM instance from config"""
        provider = LLMProvider(config["provider"])
        model_name = config["model_name"]
        api_key = config.get("api_key")
        kwargs = config.get("kwargs", {})
        
        return LLMFactory.create_llm(provider, model_name, api_key, **kwargs)


class LLMManager:
    """LLM manager supporting multiple LLM instances"""
    
    def __init__(self, log_file: str = "llm_responses.log"):
        self.llms: Dict[str, BaseLLMInterface] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_file = log_file
        # Clear old log file on initialization
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("")
    
    def add_llm(self, name: str, llm: BaseLLMInterface):
        """Add LLM instance"""
        self.llms[name] = llm
        self.logger.info(f"Added LLM: {name} ({llm.get_provider().value})")
    
    def get_llm(self, name: str) -> BaseLLMInterface:
        """Get LLM instance"""
        if name not in self.llms:
            raise ValueError(f"LLM not found: {name}")
        return self.llms[name]
    
    def list_llms(self) -> List[str]:
        """List all LLM names"""
        return list(self.llms.keys())
    
    async def generate_response(self, llm_name: str, prompt: str, **kwargs) -> LLMResponse:
        """Generate response"""
        llm = self.get_llm(llm_name)
        response = await llm.generate_response(prompt, **kwargs)
        
        if not llm.validate_response(response):
            self.logger.warning(f"Invalid response from {llm_name}: {response.content}")
        
        self._log_prompt_response(llm_name, prompt, response.content)
        
        return response

    def _log_prompt_response(self, llm_name: str, prompt: str, response: str):
        """Log prompt and response to file"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"LLM: {llm_name}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response}\n")
            f.write("="*40 + "\n")
    
    def get_usage_stats(self) -> Dict[str, Dict]:
        """Get usage statistics"""
        stats = {}
        for name, llm in self.llms.items():
            stats[name] = {
                "provider": llm.get_provider().value,
                "model": llm.model_name
            }
        return stats
