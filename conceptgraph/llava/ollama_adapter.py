"""
Ollama adapter to replace LLaVA functionality in ConceptGraphs
"""

import base64
import json
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch


class OllamaVisionChat:
    """Adapter class to use Ollama vision models as a drop-in replacement for LLaVA"""
    
    def __init__(self, model_name="llama3.2-vision:latest", base_url="http://localhost:11434"):
        """
        Initialize Ollama vision chat
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2-vision:latest")
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test if Ollama is running and model is available
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama server is running and model is loaded"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name not in model_names:
                    print(f"Warning: Model {self.model_name} not found in available models: {model_names}")
                else:
                    print(f"Ollama connection successful. Using model: {self.model_name}")
            else:
                raise ConnectionError(f"Ollama server responded with status {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama server at {self.base_url}: {e}")
    
    def reset(self):
        """Reset conversation state (for compatibility with LLaVA interface)"""
        # Ollama is stateless, so nothing to reset
        pass
    
    def _image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        if isinstance(image, Image.Image):
            # Convert PIL Image to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_data = buffer.getvalue()
            return base64.b64encode(img_data).decode('utf-8')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL then to base64
            pil_image = Image.fromarray(image.astype(np.uint8))
            return self._image_to_base64(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def __call__(self, query, image_features=None, image=None):
        """
        Main interface method compatible with LLaVA
        
        Args:
            query: Text query/prompt
            image_features: Ignored (for LLaVA compatibility)
            image: PIL Image or numpy array (actual image data)
            
        Returns:
            str: Generated text response
        """
        if image is None:
            # Text-only query
            return self._generate_text(query)
        else:
            # Vision + text query
            return self._generate_vision(query, image)
    
    def _generate_text(self, prompt):
        """Generate text response without image"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            print(f"Error in Ollama text generation: {e}")
            return f"Error: {str(e)}"
    
    def _generate_vision(self, prompt, image):
        """Generate response with image and text"""
        # Convert image to base64
        image_b64 = self._image_to_base64(image)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
        
        # Try multiple times with shorter timeout
        for attempt in range(3):
            try:
                timeout = 60 if attempt == 0 else 30  # First attempt longer timeout
                response = requests.post(self.api_url, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                return result.get('response', '').strip()
            except Exception as e:
                print(f"Attempt {attempt + 1}/3 failed: {e}")
                if attempt == 2:  # Last attempt
                    return "small object"  # Return fallback description
                continue
    
    def load_image(self, image_file):
        """Load image from file path (for compatibility with LLaVA interface)"""
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
    
    def encode_image(self, image_tensor):
        """
        Dummy method for LLaVA compatibility. 
        Returns the input as Ollama doesn't need pre-encoded features.
        """
        return image_tensor


# Factory function for easy integration
def create_ollama_chat(model_name="llama3.2-vision:latest"):
    """Create OllamaVisionChat instance"""
    return OllamaVisionChat(model_name=model_name)


if __name__ == "__main__":
    # Test the adapter
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="llama3.2-vision:latest")
    args = parser.parse_args()
    
    # Initialize chat
    chat = create_ollama_chat(args.model_name)
    print(f"Ollama chat initialized with model: {args.model_name}")
    
    # Load and test image
    image = chat.load_image(args.image_file)
    
    # Test vision query
    query = "Describe the central object in the image."
    response = chat(query=query, image=image)
    print(f"Query: {query}")
    print(f"Response: {response}")