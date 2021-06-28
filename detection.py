import json


class Detection:

    def __init__(self, title, confidence, location):
        self.title = title
        self.confidence = confidence
        self.location = location

    def get_json_string(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return f'{self.title}, {self.confidence}, {self.location}'
