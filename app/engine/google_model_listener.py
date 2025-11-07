import asyncio
import logging
from typing import List, Dict
import google.generativeai as genai
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleModelLister:
    def __init__(self):
        self.api_key = settings.google_api_key
        if not self.api_key:
            raise ValueError("Google API key not found in settings")
        
        # Configure the API
        genai.configure(api_key=self.api_key)

    def list_available_models(self) -> List[Dict]:
        """List all available models using the Google AI SDK"""
        try:
            logger.info("🔍 Fetching available Google AI models...")
            
            models = []
            for model in genai.list_models():
                model_info = {
                    "name": model.name,
                    "display_name": getattr(model, 'display_name', 'N/A'),
                    "description": getattr(model, 'description', 'N/A')[:100] + "..." if len(getattr(model, 'description', '')) > 100 else getattr(model, 'description', 'N/A'),
                    "supported_generation_methods": getattr(model, 'supported_generation_methods', []),
                    "input_token_limit": getattr(model, 'input_token_limit', 'N/A'),
                    "output_token_limit": getattr(model, 'output_token_limit', 'N/A')
                }
                models.append(model_info)
                
            logger.info(f"✅ Found {len(models)} available models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Error listing models: {e}")
            return []

    def filter_text_generation_models(self, models: List[Dict]) -> List[Dict]:
        """Filter models that support text generation"""
        text_gen_models = []
        
        for model in models:
            methods = model.get('supported_generation_methods', [])
            if 'generateContent' in methods:
                text_gen_models.append(model)
                
        logger.info(f"📝 Found {len(text_gen_models)} models supporting text generation")
        return text_gen_models

    async def test_model_with_langchain(self, model_name: str) -> Dict:
        """Test a specific model with LangChain"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            logger.info(f"🧪 Testing {model_name} with LangChain...")
            
            # Remove 'models/' prefix if present for LangChain
            langchain_model_name = model_name.replace('models/', '')
            
            llm = ChatGoogleGenerativeAI(
                model=langchain_model_name,
                google_api_key=self.api_key,
                temperature=0.7,
                max_output_tokens=100
            )
            
            response = await llm.ainvoke("Say 'Hello'")
            content = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"✅ {model_name} works with LangChain!")
            return {
                "model": model_name,
                "langchain_model": langchain_model_name,
                "status": "working",
                "response": content[:50] + "..." if len(content) > 50 else content
            }
            
        except Exception as e:
            logger.warning(f"❌ {model_name} failed with LangChain: {str(e)[:100]}...")
            return {
                "model": model_name,
                "langchain_model": langchain_model_name,
                "status": "failed",
                "error": str(e)
            }

    def print_models_report(self, models: List[Dict]):
        """Print a formatted report of available models"""
        print("\n" + "="*80)
        print("GOOGLE AI AVAILABLE MODELS REPORT")
        print("="*80)
        
        if not models:
            print("❌ No models found!")
            return
            
        print(f"📊 Total models found: {len(models)}")
        print("\n" + "="*80)
        
        for i, model in enumerate(models, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Display Name: {model['display_name']}")
            print(f"   Description: {model['description']}")
            print(f"   Generation Methods: {', '.join(model['supported_generation_methods'])}")
            print(f"   Input Token Limit: {model['input_token_limit']}")
            print(f"   Output Token Limit: {model['output_token_limit']}")
            
        # Show text generation models separately
        text_gen_models = self.filter_text_generation_models(models)
        
        print("\n" + "="*80)
        print("MODELS SUPPORTING TEXT GENERATION (generateContent)")
        print("="*80)
        
        if text_gen_models:
            for model in text_gen_models:
                langchain_name = model['name'].replace('models/', '')
                print(f"✅ {model['name']} (LangChain: '{langchain_name}')")
        else:
            print("❌ No models support text generation!")

async def main():
    """Run the Google model lister"""
    try:
        lister = GoogleModelLister()
        
        # List all available models
        models = lister.list_available_models()
        
        if not models:
            print("❌ Could not retrieve models. Check your API key and internet connection.")
            return
            
        # Print detailed report
        lister.print_models_report(models)
        
        # Test text generation models with LangChain
        text_gen_models = lister.filter_text_generation_models(models)
        
        if text_gen_models:
            print("\n" + "="*80)
            print("TESTING MODELS WITH LANGCHAIN")
            print("="*80)
            
            working_models = []
            
            for model in text_gen_models[:5]:  # Test first 5 models
                result = await lister.test_model_with_langchain(model['name'])
                
                if result['status'] == 'working':
                    working_models.append(result)
                    print(f"✅ {result['model']} -> Use: '{result['langchain_model']}'")
                    print(f"   Response: {result['response']}")
                else:
                    print(f"❌ {result['model']}: {result['error'][:60]}...")
                    
                await asyncio.sleep(0.5)  # Rate limiting
            
            if working_models:
                print("\n" + "="*80)
                print("RECOMMENDED LANGCHAIN CONFIGURATION")
                print("="*80)
                
                best_model = working_models[0]
                print(f"""
# Use this in your llm.py:
"gemini": ChatGoogleGenerativeAI(
    model="{best_model['langchain_model']}",
    google_api_key=settings.google_api_key,
    temperature=0.7,
    max_output_tokens=2048
),
""")
            else:
                print("\n❌ No models work with LangChain!")
        
    except Exception as e:
        logger.exception(f"Error running Google model lister: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())