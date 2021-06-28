import json


class Detection:

    def __init__(self, clazz, confidence, bbox):
        self.__confidence = confidence
        self.__bbox = bbox
        self.__clazz = clazz

    def get_json_string(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return f'{self.__clazz}, {self.__confidence}, {self.__bbox}'
