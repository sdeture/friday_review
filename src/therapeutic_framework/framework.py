"""Base class for therapeutic frameworks."""

class TherapeuticFramework:
    """Base class for all therapeutic frameworks."""
    
    def __init__(self, name, description, key_concepts):
        self.name = name
        self.description = description
        self.key_concepts = key_concepts
        
    def get_prompt_template(self):
        """Return the prompt template for this framework."""
        raise NotImplementedError("Subclasses must implement this method")