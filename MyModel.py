
from sklearn.externals import joblib
import connect_s3

class MyModel(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """
    def __init__(self, bucket, model_key):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        print("bucket: " + bucket)
        print("model key: " + model_key)
        self.loaded = False
        self.model = None
        self.s3 = connect_s3.ConnectS3(bucket, model_key)

    def load(self):
        self.model = joblib.load("./model.m")
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
            return "less is more more more more"


# aa = MyModel("kubeflow-anonymous-test", 'test/2019-12-16T17-43-41Z/test')
# print(aa.predict([[5,5,3],[2,2,2]]))
