class BaseTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, *args, **kwargs):
        pass

    @classmethod
    def from_function(cls, fn, name=None, description=None):
        if name is None:
            name = fn.__name__
        if description is None:
            description = fn.__doc__

        tool_from_fn = cls(name, description)
        tool_from_fn.run = fn
        return tool_from_fn
