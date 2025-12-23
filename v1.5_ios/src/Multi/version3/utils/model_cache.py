# Model Cache
from typing import Dict, Any, Callable, Optional

class ModelCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get_or_load(self, key: str, load_fn: Callable) -> Any:
        if key not in self._cache:
            print(f"  🔄 Loading {key}...")
            self._cache[key] = load_fn()
        else:
            print(f"  ♻️  Using cached {key}")
        return self._cache[key]
    
    def clear(self, key: Optional[str] = None):
        if key is None:
            self._cache.clear()
        elif key in self._cache:
            del self._cache[key]

model_cache = ModelCache()
