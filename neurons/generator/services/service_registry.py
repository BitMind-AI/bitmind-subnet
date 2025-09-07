from typing import Dict, List, Optional, Any
import bittensor as bt

from .base_service import BaseGenerationService
from .openai_service import OpenAIService
from .openrouter_service import OpenRouterService
from .local_service import LocalService


class ServiceRegistry:
    """
    Registry for managing different generation services.
    
    The registry automatically discovers available services and routes
    tasks to the best available service for each task type.
    """
    
    def __init__(self, config: Any = None):
        self.config = config
        self.services: List[BaseGenerationService] = []
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all available services."""
        bt.logging.info("Initializing generation services...")
        
        # Try to initialize services in order of preference
        service_classes = [
            OpenAIService,      # 3rd party API service
            OpenRouterService,  # OpenRouter API service
            LocalService,       # Local model service
        ]
        
        for service_class in service_classes:
            try:
                service = service_class(self.config)
                if service.is_available():
                    self.services.append(service)
                    bt.logging.info(f"✅ {service.name} is available")
                else:
                    bt.logging.info(f"❌ {service.name} is not available")
            except Exception as e:
                bt.logging.warning(f"Failed to initialize {service_class.__name__}: {e}")
        
        bt.logging.info(f"Initialized {len(self.services)} generation services")
    
    def get_service(self, modality: str) -> Optional[BaseGenerationService]:
        """Get the best available service for a modality."""
        for service in self.services:
            if service.supports_modality(modality):
                bt.logging.debug(f"Using {service.name} for modality={modality}")
                return service
        
        bt.logging.warning(f"No service available for modality={modality}")
        return None
    
    def get_available_services(self) -> List[Dict[str, Any]]:
        """Get information about all available services."""
        return [service.get_info() for service in self.services]
    
    def get_all_api_key_requirements(self) -> Dict[str, str]:
        """
        Get API key requirements from all services (both available and unavailable).
        
        This method creates instances of all service classes to get their API key
        requirements, regardless of whether they're currently available.
        
        Returns:
            Dict with environment variable names as keys and descriptions as values.
        """
        all_requirements = {}
        
        service_classes = [
            OpenAIService,
            OpenRouterService,
            LocalService,
        ]
        
        for service_class in service_classes:
            try:
                # Create a temporary instance to get API key requirements
                # We don't need it to be fully initialized, just to call the method
                temp_service = service_class()
                requirements = temp_service.get_api_key_requirements()
                all_requirements.update(requirements)
            except Exception as e:
                bt.logging.warning(f"Failed to get API key requirements from {service_class.__name__}: {e}")
        
        return all_requirements
    
    def reload_services(self):
        """Reload all services (useful for configuration changes)."""
        self.services.clear()
        self._initialize_services()
