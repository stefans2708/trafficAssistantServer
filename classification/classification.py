class Classification:

    def __init__(self, title, confidence):
        self.title = title
        self.confidence = confidence

    def __repr__(self):
        return f'{self.title}, {self.confidence}'
