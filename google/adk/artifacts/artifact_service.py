class InMemoryArtifactService:
    def __init__(self):
        self._store = {}

    def save(self, key: str, value):
        self._store[key] = value

    def get(self, key: str):
        return self._store.get(key)
