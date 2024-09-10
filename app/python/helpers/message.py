class Message:
    def __init__(self, content):
        self.content = content


class HumanMessage(Message):
    pass


class SystemMessage(Message):
    pass


class AIMessage(Message):
    pass
