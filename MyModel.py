
from sklearn.externals import joblib

class MyModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """
    def __init__(self, model_path, fix = 2):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        print("model_path: " + model_path)
        self.fix = fix
        self.model_path = model_path
        self.loaded = False
        self.model = None

    def load(self):
        self.model = joblib.load(self.model_path)
        self.loaded = True

    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        print(X)
        if not self.loaded:
            self.load()

        if self.model:
            return self.model.predict(X)
        else:
            return "less is more more more more %d" % self.fix


#aa = MyModel(model_path='/Users/jinjianbing/Downloads/model.m')
#print(aa.predict([[5,5,3],[2,2,2]]))
