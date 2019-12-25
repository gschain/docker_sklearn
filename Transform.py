
class Transform(object):
    __x = None

    def transform_input(self, X):
        self.__x = X

    def transform_output(self, model):
        return model.predict(self.__x)