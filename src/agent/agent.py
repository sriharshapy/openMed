#!/usr/bin/env python3
"""
OpenAI-powered Agentic Workflow for Multi-Disease Medical Intent Classification
This module provides an intelligent agent that uses OpenAI's API to analyze user prompts 
and determine if they want to perform medical analysis for pneumonia, brain tumor, or 
tuberculosis detection from medical images.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import asyncio

# Third party imports
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Enumeration of possible intent types"""
    PNEUMONIA_CHECK = "pneumonia_check"
    BRAIN_TUMOR_CHECK = "brain_tumor_check" 
    TB_CHECK = "tb_check"
    GENERAL_MEDICAL = "general_medical" 
    MEDICAL_IMAGING = "medical_imaging"
    OTHER = "other"


class DiseaseType(Enum):
    """Enumeration of supported disease types"""
    PNEUMONIA = "pneumonia"
    BRAIN_TUMOR = "brain_tumor"
    TB = "tb"
    TUBERCULOSIS = "tuberculosis"  # alias for TB


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels"""
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"        # < 0.5


class OpenAIMultiDiseaseAgent:
    """
    OpenAI-powered agent for analyzing user prompts to detect multi-disease check intent.
    Uses OpenAI's language models for sophisticated natural language understanding.
    Supports: Pneumonia, Brain Tumor, and Tuberculosis detection.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI agent
        
        Args:
            api_key (str, optional): OpenAI API key. If None, will try to load from environment
            model (str): OpenAI model to use for analysis
        """
        self.model = model
        
        # Load API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('OPENAI_KEY') or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_KEY in your .env file or pass it directly."
            )
        
        # Initialize OpenAI client with new API
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Define the system prompt for multi-disease intent detection
        self.system_prompt = """You are a medical AI assistant specialized in analyzing user prompts to determine if they want to perform medical image analysis for disease detection.

You can detect intent for three types of medical analysis:
1. PNEUMONIA detection from chest X-rays
2. BRAIN TUMOR detection from brain MRI scans
3. TUBERCULOSIS (TB) detection from chest X-rays

Your task is to analyze user input and determine:
1. Which disease they want to check for (if any)
2. Whether they actually want medical analysis (boolean)
3. The confidence level of your assessment (0.0 to 1.0)
4. The specific intent type
5. A brief reasoning for your decision
6. Always cater to the last user message.

You must respond with a valid JSON object in this exact format:
{
    "wants_medical_analysis": boolean,
    "disease_type": "pneumonia" | "brain_tumor" | "tb" | null,
    "confidence": float (0.0 to 1.0),
    "intent_type": "pneumonia_check" | "brain_tumor_check" | "tb_check" | "medical_imaging" | "general_medical" | "other",
    "reasoning": "brief explanation of your decision"
}

Guidelines for disease detection:

PNEUMONIA (chest X-rays):
- Keywords: pneumonia, lung infection, chest X-ray, respiratory infection, pulmonary consolidation
- Image types: chest radiograph, thoracic imaging, lung scan
- Symptoms mentioned: cough, fever, difficulty breathing, chest pain

BRAIN TUMOR (MRI scans):
- Keywords: brain tumor, brain cancer, glioma, meningioma, cranial mass, brain lesion
- Image types: brain MRI, cranial scan, neuroimaging, brain scan
- Symptoms mentioned: headaches, seizures, neurological symptoms, vision problems

TUBERCULOSIS/TB (chest X-rays):
- Keywords: tuberculosis, TB, pulmonary TB, mycobacterium, lung TB
- Image types: chest X-ray, thoracic imaging, lung scan
- Symptoms mentioned: persistent cough, night sweats, weight loss, hemoptysis

wants_medical_analysis should be TRUE if:
- User asks to analyze/diagnose any of the three diseases from medical images
- They want to check/detect/screen for these conditions
- They mention uploading medical scans for analysis
- They ask about abnormalities in relevant medical images

wants_medical_analysis should be FALSE if:
- General health questions without imaging
- Non-medical topics
- They state images are "normal" or "clear" 
- Educational/research questions without actual analysis

Intent types:
- "pneumonia_check": Direct pneumonia detection request
- "brain_tumor_check": Direct brain tumor detection request  
- "tb_check": Direct tuberculosis detection request
- "medical_imaging": General medical imaging request (unclear which disease)
- "general_medical": Medical question but not imaging-specific
- "other": Non-medical query

Confidence levels:
- 0.9-1.0: Very clear medical imaging request with specific disease keywords
- 0.7-0.89: Clear medical context with relevant imaging type
- 0.5-0.69: Some medical indicators but ambiguous disease type
- 0.3-0.49: Weak medical context
- 0.0-0.29: No clear medical intent

Remember: You must respond with ONLY the JSON object, no additional text."""

        logger.info(f"OpenAI Multi-Disease Agent initialized with model: {model}")
    
    async def analyze_prompt_async(self, prompt: str) -> Dict[str, Any]:
        """
        Asynchronously analyze a user prompt using OpenAI API
        
        Args:
            prompt (str): User input to analyze
            
        Returns:
            Dict[str, Any]: Analysis results in JSON format
        """
        return self.analyze_prompt(prompt)
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a user prompt using OpenAI API
        
        Args:
            prompt (str): User input to analyze
            
        Returns:
            Dict[str, Any]: Analysis results in JSON format
        """
        
        if not prompt or not prompt.strip():
            return {
                "wants_medical_analysis": False,
                "disease_type": None,
                "confidence": 0.0,
                "confidence_level": "low",
                "intent_type": "other",
                "reasoning": "Empty or invalid input",
                "original_prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "agent_version": "3.0-openai-multi-disease",
                "model_used": self.model
            }
        
        try:
            # Make OpenAI API call with structured output
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            
            # Extract the response content
            ai_response = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            try:
                parsed_result = json.loads(ai_response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {ai_response}")
                return self._create_error_response(prompt, f"JSON parsing error: {str(e)}")
            
            # Validate and enhance the response
            enhanced_result = self._validate_and_enhance_response(parsed_result, prompt)
            
            logger.info(f"Analyzed prompt: '{prompt[:50]}...' -> wants_medical_analysis: {enhanced_result['wants_medical_analysis']}, disease: {enhanced_result['disease_type']}")
            return enhanced_result
            
        except Exception as e:
            # Handle all OpenAI API errors with the new client
            error_type = type(e).__name__
            if "rate_limit" in str(e).lower():
                logger.error("OpenAI API rate limit exceeded")
                return self._create_error_response(prompt, "API rate limit exceeded")
            elif "invalid_request" in str(e).lower():
                logger.error(f"Invalid OpenAI API request: {e}")
                return self._create_error_response(prompt, f"Invalid API request: {str(e)}")
            else:
                logger.error(f"OpenAI API error ({error_type}): {e}")
                return self._create_error_response(prompt, f"API error: {str(e)}")
    
    def _validate_and_enhance_response(self, ai_result: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        Validate and enhance the AI response with additional metadata
        
        Args:
            ai_result (Dict[str, Any]): Raw response from OpenAI
            prompt (str): Original user prompt
            
        Returns:
            Dict[str, Any]: Enhanced response with validation and metadata
        """
        
        # Set defaults for any missing fields
        enhanced_result = {
            "wants_medical_analysis": ai_result.get("wants_medical_analysis", False),
            "disease_type": ai_result.get("disease_type", None),
            "confidence": float(ai_result.get("confidence", 0.0)),
            "intent_type": ai_result.get("intent_type", "other"),
            "reasoning": ai_result.get("reasoning", "No reasoning provided"),
            "original_prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "agent_version": "3.0-openai-multi-disease",
            "model_used": self.model
        }
        
        # Validate confidence range
        if enhanced_result["confidence"] < 0.0:
            enhanced_result["confidence"] = 0.0
        elif enhanced_result["confidence"] > 1.0:
            enhanced_result["confidence"] = 1.0
        
        # Determine confidence level
        if enhanced_result["confidence"] >= 0.8:
            enhanced_result["confidence_level"] = "high"
        elif enhanced_result["confidence"] >= 0.5:
            enhanced_result["confidence_level"] = "medium"  
        else:
            enhanced_result["confidence_level"] = "low"
        
        # Validate disease_type
        valid_diseases = ["pneumonia", "brain_tumor", "tb", None]
        if enhanced_result["disease_type"] not in valid_diseases:
            # Handle tuberculosis alias
            if enhanced_result["disease_type"] == "tuberculosis":
                enhanced_result["disease_type"] = "tb"
            else:
                enhanced_result["disease_type"] = None
        
        # Validate intent_type
        valid_intents = ["pneumonia_check", "brain_tumor_check", "tb_check", "medical_imaging", "general_medical", "other"]
        if enhanced_result["intent_type"] not in valid_intents:
            enhanced_result["intent_type"] = "other"
        
        # Consistency checks
        if enhanced_result["wants_medical_analysis"] and not enhanced_result["disease_type"]:
            # If they want medical analysis but no specific disease, set to medical_imaging
            if enhanced_result["intent_type"] in ["pneumonia_check", "brain_tumor_check", "tb_check"]:
                enhanced_result["intent_type"] = "medical_imaging"
        
        # Add disease-specific metadata
        if enhanced_result["disease_type"]:
            enhanced_result["disease_info"] = self._get_disease_info(enhanced_result["disease_type"])
        
        return enhanced_result
    
    def _get_disease_info(self, disease_type: str) -> Dict[str, Any]:
        """Get metadata for a specific disease type."""
        disease_info = {
            "pneumonia": {
                "full_name": "Pneumonia",
                "image_type": "chest_xray",
                "classes": ["Normal", "Pneumonia"],
                "num_classes": 2,
                "api_endpoint": "/predict/pneumonia"
            },
            "brain_tumor": {
                "full_name": "Brain Tumor",  
                "image_type": "brain_mri",
                "classes": ["Glioma", "Meningioma", "Tumor"],
                "num_classes": 3,
                "api_endpoint": "/predict/brain_tumor"
            },
            "tb": {
                "full_name": "Tuberculosis",
                "image_type": "chest_xray", 
                "classes": ["Normal", "TB"],
                "num_classes": 2,
                "api_endpoint": "/predict/tb"
            }
        }
        
        return disease_info.get(disease_type, {})
    
    def _create_error_response(self, prompt: str, error_message: str) -> Dict[str, Any]:
        """
        Create a standardized error response
        
        Args:
            prompt (str): Original user prompt
            error_message (str): Error description
            
        Returns:
            Dict[str, Any]: Standardized error response
        """
        return {
            "wants_medical_analysis": False,
            "disease_type": None,
            "confidence": 0.0,
            "confidence_level": "low",
            "intent_type": "other",
            "reasoning": f"Error occurred: {error_message}",
            "original_prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "agent_version": "3.0-openai-multi-disease",
            "model_used": self.model,
            "error": True,
            "error_message": error_message
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_medical_intent(prompt: str) -> Dict[str, Any]:
    """
    Convenience function to analyze medical intent for multiple diseases.
    
    Args:
        prompt (str): User input to analyze
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    try:
        agent = OpenAIMultiDiseaseAgent()
        return agent.analyze_prompt(prompt)
    except Exception as e:
        logger.error(f"Error in analyze_medical_intent: {str(e)}")
        return {
            "wants_medical_analysis": False,
            "disease_type": None,
            "confidence": 0.0,
            "confidence_level": "low",
            "intent_type": "other",
            "reasoning": f"Agent initialization failed: {str(e)}",
            "original_prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "agent_version": "3.0-openai-multi-disease",
            "error": True,
            "error_message": str(e)
        }

def demo_analysis():
    """
    Demonstrate the multi-disease analysis functionality with example prompts.
    """
    print("🔬 Multi-Disease Medical Intent Analysis Demo")
    print("=" * 60)
    
    # Test prompts for different diseases
    test_prompts = [
        # Pneumonia tests
        "Can you check this chest X-ray for pneumonia?",
        "I have a lung infection, can you analyze my chest radiograph?",
        "Please detect pneumonia from this medical image",
        
        # Brain tumor tests  
        "Can you analyze this brain MRI for tumors?",
        "I need to check for brain cancer in this scan",
        "Please detect glioma or meningioma in this brain image",
        
        # TB tests
        "Can you check this chest X-ray for tuberculosis?", 
        "I need TB detection from this lung scan",
        "Please analyze this image for pulmonary tuberculosis",
        
        # General/ambiguous
        "Can you analyze this medical scan?",
        "I have a medical image to check",
        "What do you think about this X-ray?",
        
        # Non-medical
        "What's the weather like today?",
        "How do I cook pasta?",
        "Tell me about machine learning"
    ]
    
    try:
        agent = OpenAIMultiDiseaseAgent()
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i:2d}. Prompt: '{prompt}'")
            result = agent.analyze_prompt(prompt)
            
            print(f"    → Medical Analysis: {result['wants_medical_analysis']}")
            print(f"    → Disease Type: {result['disease_type']}")
            print(f"    → Intent: {result['intent_type']}")
            print(f"    → Confidence: {result['confidence']:.2f} ({result['confidence_level']})")
            print(f"    → Reasoning: {result['reasoning']}")
            
            if result.get('disease_info'):
                info = result['disease_info']
                print(f"    → API Endpoint: {info['api_endpoint']}")
                print(f"    → Classes: {info['classes']}")
    
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("Make sure you have OPENAI_KEY set in your .env file")

def test_api_key():
    """Test if OpenAI API key is properly configured."""
    try:
        agent = OpenAIMultiDiseaseAgent()
        test_result = agent.analyze_prompt("test")
        print("✅ OpenAI API key is working")
        return True
    except Exception as e:
        print(f"❌ OpenAI API key test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run demo when script is executed directly
    demo_analysis()
