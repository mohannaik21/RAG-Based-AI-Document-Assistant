import os
import requests
import time
from dotenv import load_dotenv
import streamlit as st
import json

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
API_URL = "https://api.mistral.ai/v1/chat/completions"

class EnhancedQAEngine:
    """Enhanced Question-Answering engine with better prompting and error handling"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.api_url = API_URL
        self.max_retries = 3
        self.retry_delay = 1
    
    def generate_answer(self, context, question, conversation_history=None):
        """
        Generate an answer using Mistral AI with enhanced prompting
        
        Args:
            context: Relevant text chunks
            question: User's question
            conversation_history: Previous conversation for context
        
        Returns:
            Generated answer string
        """
        try:
            # Validate inputs
            if not context.strip():
                return "❌ No relevant context found to answer your question."
            
            if not question.strip():
                return "❌ Please provide a valid question."
            
            # Create enhanced prompt
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(context, question, conversation_history)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add conversation history if available
            if conversation_history:
                messages = self._add_conversation_history(messages, conversation_history)
            
            # Generate response with retry logic
            response = self._make_api_call(messages)
            
            if response:
                return self._post_process_answer(response)
            else:
                return "❌ Failed to generate response. Please try again."
                
        except Exception as e:
            st.error(f"Error in QA engine: {str(e)}")
            return f"❌ Error generating answer: {str(e)}"
    
    def _create_system_prompt(self):
        """Create an enhanced system prompt"""
        return """You are an intelligent AI assistant specializing in document analysis and question answering. Your role is to provide accurate, helpful, and well-structured answers based on the given context.

Guidelines for your responses:
1. **Accuracy**: Only use information from the provided context. If the answer isn't in the context, clearly state this.
2. **Clarity**: Provide clear, well-structured answers that are easy to understand.
3. **Completeness**: Give comprehensive answers that fully address the question.
4. **Citations**: When possible, reference specific parts of the context.
5. **Honesty**: If you're uncertain or if information is unclear, acknowledge this.
6. **Formatting**: Use proper formatting with bullet points, numbers, or paragraphs as appropriate.

Response Structure:
- Start with a direct answer to the question
- Provide supporting details from the context
- Include relevant examples or explanations
- End with any caveats or limitations if applicable

Remember: You are answering based on document content, so maintain an informative and professional tone."""
    
    def _create_user_prompt(self, context, question, conversation_history=None):
        """Create an enhanced user prompt with context and question"""
        
        # Truncate context if too long (keep within token limits)
        max_context_length = 3000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n[... content truncated ...]"
        
        prompt = f"""Based on the following document content, please answer the question comprehensively.

**Document Content:**
{context}

**Question:** {question}

**Instructions:**
- Use only the information provided in the document content above
- If the answer requires information not in the context, clearly state this limitation
- Provide specific details and examples from the text when possible
- Structure your answer clearly with appropriate formatting
- If multiple aspects of the question can be answered, address each one"""

        return prompt
    
    def _add_conversation_history(self, messages, history):
        """Add relevant conversation history for context"""
        if not history:
            return messages
        
        # Add last few exchanges for context (limit to avoid token overflow)
        recent_history = history[-3:] if len(history) > 3 else history
        
        for question, answer, _, _ in recent_history:
            messages.insert(-1, {"role": "user", "content": f"Previous Question: {question}"})
            messages.insert(-1, {"role": "assistant", "content": f"Previous Answer: {answer[:200]}..."})
        
        return messages
    
    def _make_api_call(self, messages):
        """Make API call with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-small",
            "messages": messages,
            "temperature": 0.3,  # Lower temperature for more consistent answers
            "max_tokens": 1500,
            "top_p": 0.9
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    st.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 401:
                    return "❌ Invalid API key. Please check your Mistral API configuration."
                
                else:
                    error_msg = f"API Error {response.status_code}"
                    try:
                        error_detail = response.json().get("message", response.text)
                        error_msg += f": {error_detail}"
                    except:
                        error_msg += f": {response.text[:200]}"
                    
                    if attempt < self.max_retries - 1:
                        st.warning(f"{error_msg}. Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return f"❌ {error_msg}"
            
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    st.warning(f"Request timeout. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return "❌ Request timeout. Please try again."
            
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    st.warning(f"Connection error. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return "❌ Connection error. Please check your internet connection."
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    st.warning(f"Unexpected error: {str(e)}. Retrying...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return f"❌ Unexpected error: {str(e)}"
        
        return None
    
    def _post_process_answer(self, answer):
        """Post-process the generated answer"""
        if not answer:
            return "❌ No response generated."
        
        # Clean up common formatting issues
        answer = answer.strip()
        
        # Ensure proper paragraph breaks
        answer = answer.replace('\n\n\n', '\n\n')
        
        # Add helpful prefixes for clarity
        if not answer.startswith(('❌', '✅', '📝', '💡', '⚠️')):
            if 'cannot' in answer.lower() or 'not found' in answer.lower():
                answer = f"⚠️ {answer}"
            else:
                answer = f"📝 {answer}"
        
        return answer
    
    def validate_api_key(self):
        """Validate the Mistral API key"""
        if not self.api_key:
            return False, "No API key found. Please add MISTRAL_API_KEY to your .env file."
        
        # Test API call
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_payload = {
            "model": "mistral-small",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "API key is valid."
            elif response.status_code == 401:
                return False, "Invalid API key."
            else:
                return False, f"API test failed with status {response.status_code}"
                
        except Exception as e:
            return False, f"API test failed: {str(e)}"

# Global instance
qa_engine = EnhancedQAEngine()

def generate_answer(context, question, conversation_history=None):
    """
    Wrapper function for backward compatibility
    """
    return qa_engine.generate_answer(context, question, conversation_history)

def validate_api_configuration():
    """Validate API configuration"""
    return qa_engine.validate_api_key()

def get_model_info():
    """Get information about the current model"""
    return {
        "model_name": "Mistral Small",
        "provider": "Mistral AI",
        "max_tokens": 1500,
        "temperature": 0.3,
        "features": [
            "Document Q&A",
            "Context-aware responses",
            "Multi-turn conversations",
            "Error handling & retries",
            "Professional formatting"
        ]
    }