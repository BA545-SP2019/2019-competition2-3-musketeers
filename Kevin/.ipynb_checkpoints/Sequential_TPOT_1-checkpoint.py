import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, PolynomialFeatures

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('Sequential_Data1.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('Y', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Y'].values, random_state=2019)

# Average CV score on the training set was:0.49510757579372305
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    Binarizer(threshold=0.8),
    BernoulliNB(alpha=10.0, fit_prior=True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
