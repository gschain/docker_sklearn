import urllib.request
import pickle
from sklearn.externals import joblib
import pandas as pd


class MyModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """
    def __init__(self, fix = 2, url = 'https://shield.mlamp.cn/task/api/file/space/download/147cbad2739812c9973c8725bac26552/60288/model.m'):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        print("url: " + url)
        self.fix = fix
        self.url = url
        self.loaded = False
        self.model = None

    def load(self):
        urllib.request.urlretrieve(self.url, "model.m")
        self.model = joblib.load('model.m')
        self.loaded = True

    def predict(self, X, features_names=None):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        if not self.loaded:
            self.load()

        if self.model:
            return self.model.predict(X)
        else:
            return "less is more more more more %d" % self.fix


#aa = MyModel()
#print(aa.predict([[5,5,3],[2,2,2]]))
