class CustomAdapter:
    @staticmethod
    def load():
        raise NotImplementedError("Custom model adapter not implemented.")

    @staticmethod
    def generate(**kwargs):
        raise NotImplementedError("Custom model adapter not implemented.")
