import json

from classification.classification import Classification


class ClassificationResult:

    def __init__(self, classifications):
        self.classifications = classifications

    @staticmethod
    def __serialize_classification(c: Classification):
        return c.__dict__

    def serialize_to_json(self):
        return json.dumps(self.__dict__, default=self.__serialize_classification)
