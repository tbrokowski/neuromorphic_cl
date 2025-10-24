"""
Registry Utility for Neuromorphic Continual Learning.

This module provides a registry pattern for dynamically registering and
instantiating different components like encoders, models, and datasets.
"""

import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Registry:
    """
    A registry for registering and retrieving objects by name.
    
    This allows for dynamic instantiation of components without
    hard-coding imports or class names throughout the codebase.
    """
    
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}
        
    def register(self, name: str, obj: Optional[Any] = None) -> Callable:
        """
        Register an object in the registry.
        
        Can be used as a decorator or called directly.
        
        Args:
            name: Name to register under
            obj: Object to register (if not using as decorator)
            
        Returns:
            Decorator function or the registered object
        """
        def decorator(func_or_class: Any) -> Any:
            if name in self._registry:
                logger.warning(
                    f"Overriding existing registration for '{name}' in {self._name}"
                )
            
            self._registry[name] = func_or_class
            logger.debug(f"Registered '{name}' in {self._name}")
            return func_or_class
        
        if obj is not None:
            # Direct registration
            return decorator(obj)
        else:
            # Decorator usage
            return decorator
    
    def get(self, name: str) -> Any:
        """
        Get a registered object by name.
        
        Args:
            name: Name of the object to retrieve
            
        Returns:
            The registered object
            
        Raises:
            KeyError: If name is not registered
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self._name} registry. "
                f"Available: {available}"
            )
        
        return self._registry[name]
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create an instance of a registered class.
        
        Args:
            name: Name of the class to instantiate
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor
            
        Returns:
            Instance of the registered class
        """
        cls = self.get(name)
        
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create instance of '{name}': {e}")
            raise
    
    def list_available(self) -> list:
        """Get list of all registered names."""
        return list(self._registry.keys())
    
    def contains(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry
    
    def unregister(self, name: str) -> None:
        """Unregister an object."""
        if name in self._registry:
            del self._registry[name]
            logger.debug(f"Unregistered '{name}' from {self._name}")
        else:
            logger.warning(f"'{name}' not found in {self._name} for unregistration")
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._registry.clear()
        logger.debug(f"Cleared all registrations from {self._name}")
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return self.contains(name)
    
    def __len__(self) -> int:
        """Get number of registered items."""
        return len(self._registry)
    
    def __iter__(self):
        """Iterate over registered names."""
        return iter(self._registry.keys())
    
    def __getitem__(self, name: str) -> Any:
        """Support dictionary-like access."""
        return self.get(name)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Registry('{self._name}', {len(self._registry)} items)"


class LazyRegistry(Registry):
    """
    A registry that supports lazy loading of modules.
    
    This allows registering modules by import path without actually
    importing them until they're needed.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        self._import_paths: Dict[str, str] = {}
    
    def register_lazy(self, name: str, import_path: str) -> None:
        """
        Register a module by import path for lazy loading.
        
        Args:
            name: Name to register under
            import_path: Python import path (e.g., "module.submodule.Class")
        """
        self._import_paths[name] = import_path
        logger.debug(f"Registered lazy import '{name}' -> '{import_path}' in {self._name}")
    
    def get(self, name: str) -> Any:
        """
        Get a registered object, importing lazily if needed.
        
        Args:
            name: Name of the object to retrieve
            
        Returns:
            The registered object
        """
        # Check if already loaded
        if name in self._registry:
            return self._registry[name]
        
        # Check if we have a lazy import path
        if name in self._import_paths:
            import_path = self._import_paths[name]
            try:
                obj = self._import_from_path(import_path)
                self._registry[name] = obj
                return obj
            except Exception as e:
                logger.error(f"Failed to lazy import '{import_path}' for '{name}': {e}")
                raise
        
        # Not found
        available = list(self._registry.keys()) + list(self._import_paths.keys())
        raise KeyError(
            f"'{name}' not found in {self._name} registry. "
            f"Available: {available}"
        )
    
    def _import_from_path(self, import_path: str) -> Any:
        """Import an object from a dotted import path."""
        module_path, obj_name = import_path.rsplit('.', 1)
        
        try:
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, obj_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import '{import_path}': {e}")
    
    def list_available(self) -> list:
        """Get list of all registered names (including lazy)."""
        return list(set(self._registry.keys()) | set(self._import_paths.keys()))
    
    def contains(self, name: str) -> bool:
        """Check if a name is registered (including lazy)."""
        return name in self._registry or name in self._import_paths


class TypedRegistry(Registry):
    """
    A registry that enforces type constraints on registered objects.
    """
    
    def __init__(self, name: str, expected_type: Type[T]):
        super().__init__(name)
        self.expected_type = expected_type
    
    def register(self, name: str, obj: Optional[T] = None) -> Callable:
        """Register with type checking."""
        def decorator(func_or_class: T) -> T:
            # Type checking
            if not self._is_valid_type(func_or_class):
                raise TypeError(
                    f"Object '{name}' must be of type {self.expected_type}, "
                    f"got {type(func_or_class)}"
                )
            
            return super(TypedRegistry, self).register(name, func_or_class)
        
        if obj is not None:
            return decorator(obj)
        else:
            return decorator
    
    def _is_valid_type(self, obj: Any) -> bool:
        """Check if object matches expected type."""
        if isinstance(self.expected_type, type):
            # Check if it's a subclass (for classes) or instance
            if isinstance(obj, type):
                return issubclass(obj, self.expected_type)
            else:
                return isinstance(obj, self.expected_type)
        else:
            # For generic types, just do basic check
            return True


class RegistryManager:
    """
    Manager for multiple registries.
    
    Provides a centralized way to access different component registries.
    """
    
    def __init__(self):
        self._registries: Dict[str, Registry] = {}
    
    def create_registry(self, name: str, registry_type: str = "basic") -> Registry:
        """
        Create a new registry.
        
        Args:
            name: Name of the registry
            registry_type: Type of registry ("basic", "lazy", "typed")
            
        Returns:
            Created registry
        """
        if registry_type == "basic":
            registry = Registry(name)
        elif registry_type == "lazy":
            registry = LazyRegistry(name)
        elif registry_type == "typed":
            # Would need additional parameters for typed registry
            registry = Registry(name)
        else:
            raise ValueError(f"Unknown registry type: {registry_type}")
        
        self._registries[name] = registry
        return registry
    
    def get_registry(self, name: str) -> Registry:
        """Get an existing registry."""
        if name not in self._registries:
            raise KeyError(f"Registry '{name}' not found")
        return self._registries[name]
    
    def list_registries(self) -> list:
        """Get list of all registry names."""
        return list(self._registries.keys())
    
    def register_in(self, registry_name: str, object_name: str, obj: Any) -> None:
        """Register an object in a specific registry."""
        registry = self.get_registry(registry_name)
        registry.register(object_name, obj)
    
    def get_from(self, registry_name: str, object_name: str) -> Any:
        """Get an object from a specific registry."""
        registry = self.get_registry(registry_name)
        return registry.get(object_name)
    
    def create_from(self, registry_name: str, object_name: str, *args, **kwargs) -> Any:
        """Create an instance from a specific registry."""
        registry = self.get_registry(registry_name)
        return registry.create(object_name, *args, **kwargs)


# Global registry manager instance
_global_manager = RegistryManager()

# Create common registries
ENCODER_REGISTRY = _global_manager.create_registry("encoders")
MODEL_REGISTRY = _global_manager.create_registry("models")
DATASET_REGISTRY = _global_manager.create_registry("datasets")
LOSS_REGISTRY = _global_manager.create_registry("losses")
OPTIMIZER_REGISTRY = _global_manager.create_registry("optimizers")
SCHEDULER_REGISTRY = _global_manager.create_registry("schedulers")


def get_registry(name: str) -> Registry:
    """Get a registry by name."""
    return _global_manager.get_registry(name)


def register_encoder(name: str):
    """Decorator for registering encoders."""
    return ENCODER_REGISTRY.register(name)


def register_model(name: str):
    """Decorator for registering models."""
    return MODEL_REGISTRY.register(name)


def register_dataset(name: str):
    """Decorator for registering datasets."""
    return DATASET_REGISTRY.register(name)


def register_loss(name: str):
    """Decorator for registering loss functions."""
    return LOSS_REGISTRY.register(name)


def create_encoder(name: str, *args, **kwargs):
    """Create an encoder instance."""
    return ENCODER_REGISTRY.create(name, *args, **kwargs)


def create_model(name: str, *args, **kwargs):
    """Create a model instance."""
    return MODEL_REGISTRY.create(name, *args, **kwargs)


def create_dataset(name: str, *args, **kwargs):
    """Create a dataset instance."""
    return DATASET_REGISTRY.create(name, *args, **kwargs)


def list_available_encoders() -> list:
    """List all available encoders."""
    return ENCODER_REGISTRY.list_available()


def list_available_models() -> list:
    """List all available models."""
    return MODEL_REGISTRY.list_available()


def list_available_datasets() -> list:
    """List all available datasets."""
    return DATASET_REGISTRY.list_available()


# Pre-register some common components
def _register_common_components():
    """Pre-register commonly used components."""
    
    # Optimizers
    import torch.optim as optim
    OPTIMIZER_REGISTRY.register("adam", optim.Adam)
    OPTIMIZER_REGISTRY.register("adamw", optim.AdamW)
    OPTIMIZER_REGISTRY.register("sgd", optim.SGD)
    OPTIMIZER_REGISTRY.register("rmsprop", optim.RMSprop)
    
    # Schedulers
    from torch.optim import lr_scheduler
    SCHEDULER_REGISTRY.register("step", lr_scheduler.StepLR)
    SCHEDULER_REGISTRY.register("cosine", lr_scheduler.CosineAnnealingLR)
    SCHEDULER_REGISTRY.register("plateau", lr_scheduler.ReduceLROnPlateau)
    SCHEDULER_REGISTRY.register("linear", lr_scheduler.LinearLR)


# Initialize common components
_register_common_components()
