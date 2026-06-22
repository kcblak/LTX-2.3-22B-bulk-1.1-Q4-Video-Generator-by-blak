class LTXAdapter:
    @staticmethod
    def load():
        from ltx_engine import load_ltx_model
        return load_ltx_model()

    @staticmethod
    def generate(**kwargs):
        from ltx_engine import Video_Generation
        return Video_Generation(**kwargs)
