from fwk.renderers import Renderer


class Observer:

    def __init__(self) -> None:
        super().__init__()
        self.renderers = []

    def print(self, metric):
        renderer: Renderer
        for renderer in self.renderers:
            renderer.print_record(metric)

    def add_renderer(self, renderer):
        self.renderers.append(renderer)
