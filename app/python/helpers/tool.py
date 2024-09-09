class Tool:
    def __init__(self, agent):
        self.agent = agent
        self._name = None

    def execute(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self):
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, value):
        self._name = value


class Response:
    def __init__(self, message, break_loop=False):
        self.message = message
        self.break_loop = break_loop
