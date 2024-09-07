class PrintStyle:
    def __init__(
        self,
        italic=False,
        font_color=None,
        padding=False,
        background_color=None,
        bold=False,
    ):
        self.italic = italic
        self.font_color = font_color
        self.padding = padding
        self.background_color = background_color
        self.bold = bold

    def print(self, text):
        # Implement print styling logic here
        print(text)

    def stream(self, text):
        # Implement stream styling logic here
        print(text, end="", flush=True)
