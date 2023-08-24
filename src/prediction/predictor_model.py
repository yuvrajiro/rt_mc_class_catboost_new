import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the X  binary classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "catboost_multiclass_classifier"

    def __init__(
        self,
        n_estimators: Optional[int] = 423,
        learning_rate: Optional[float] = 0.16003285631484013,
        max_depth: Optional[int] = 7.0,
        auto_class_weights: Optional[str] = "SqrtBalanced",
        bagging_temperature: Optional[float] = 0.05336169179455724,
        colsample_bylevel: Optional[float] = 0.5006380634242127,
        diffusion_temperature: Optional[float] = 62.72999516347615,
        l2_leaf_reg: Optional[float] =  6.885852437148102,
        min_data_in_leaf: Optional[int] =  26.0,
        model_shrink_rate: Optional[float] = 0.022259485832457494,
        nan_mode: Optional[str] = "Forbidden",
        random_seed: Optional[int] = 2,
        boosting_type: Optional[str] = "Plain",
        loss_function: Optional[str] = "MultiClass",
        eval_metric: Optional[str] = "TotalF1",
        has_time: Optional[bool] = False,
        task_type: Optional[str] = "CPU",
        cat_features: Optional[list] = None,
        train_dir: Optional[str] = None,


        **kwargs,
    ):
        """Construct a new XGBoost binary classifier.

        Args:
            n_estimators (int, optional): The number of trees in the forest.
                Defaults to 100.
            learning_rate (float, optional): Boosting learning rate.
                Defaults to 0.1.
            max_depth (int, optional): Maximum tree depth for base learners.
                Defaults to 3.
            colsample_bytree (float, optional): Subsample ratio of columns when
                constructing each tree. Defaults to 0.8.
            min_child_weight (int, optional): Minimum sum of instance weight (hessian)
                needed in a child. Defaults to 1.
            subsample (float, optional): Subsample ratio of the training instance.
                Defaults to 0.8.
            gamma (float, optional): Minimum loss reduction required to make a further
                partition on a leaf node of the tree. Defaults to 0.
            reg_lambda (float, optional): L2 regularization term on weights.
                Defaults to 1.

        """
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.colsample_bylevel = float(colsample_bylevel)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.model_shrink_rate = float(model_shrink_rate)
        self.random_seed = int(random_seed)
        self.cat_features = cat_features
        self.boosting_type = boosting_type
        self.train_dir = train_dir
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.has_time = has_time
        self.task_type = task_type
        self.auto_class_weights = auto_class_weights
        self.bagging_temperature = bagging_temperature
        self.diffusion_temperature = diffusion_temperature
        self.l2_leaf_reg = l2_leaf_reg
        self.nan_mode = nan_mode
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> CatBoostClassifier:
        """Build a new XGBoost binary classifier."""
        if self.cat_features is None:
            self.cat_features = ['a_ct', 'a_ped_f', 'a_pedal_f', 'a_roll', 'a_hr',
                                 'a_polpur', 'day_week', 'a_dow_type', 'a_tod_type',
                                 'state', 'a_region', 'a_ru', 'a_inter', 'a_intsec',
                                 'a_roadfc', 'a_junc', 'a_relrd', 'a_ped', 'a_body',
                                 'owner', 'impact1', 'deformed', 'weather', 'lgt_cond']
        model = CatBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            colsample_bylevel=self.colsample_bylevel,
            min_data_in_leaf=self.min_data_in_leaf,
            model_shrink_rate=self.model_shrink_rate,
            cat_features=self.cat_features,
            boosting_type=self.boosting_type,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            has_time=self.has_time,
            task_type=self.task_type,
            auto_class_weights=self.auto_class_weights,
            bagging_temperature=self.bagging_temperature,
            diffusion_temperature=self.diffusion_temperature,
            l2_leaf_reg=self.l2_leaf_reg,
            nan_mode=self.nan_mode,
            random_state=self.random_seed,
            train_dir=self.train_dir,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the XGBoost binary classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """

        self.model.fit(train_inputs, train_targets,verbose=0)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the XGBoost binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the XGBoost binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the XGBoost binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the XGBoost binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded XGBoost binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        return (
            f"Model name: {self.model_name}("
            f"n_estimators: {self.n_estimators}, "
            f"learning_rate: {self.learning_rate}, "
            f"max_depth: {self.max_depth}, "
            f"colsample_bylevel: {self.colsample_bylevel}, "
            f"min_data_in_leaf: {self.min_data_in_leaf}, "
            f"model_shrink_rate: {self.model_shrink_rate}, "
            f"random_seed: {self.random_seed}, "
            f"cat_features: {self.cat_features}, "
            f"boosting_type: {self.boosting_type}, "
            f"loss_function: {self.loss_function}, "
            f"eval_metric: {self.eval_metric}, "
            f"has_time: {self.has_time}, "
            f"task_type: {self.task_type}, "
            f"auto_class_weights: {self.auto_class_weights}, "
            f"bagging_temperature: {self.bagging_temperature}, "
            f"diffusion_temperature: {self.diffusion_temperature}, "
            f"l2_leaf_reg: {self.l2_leaf_reg}, "
            f"nan_mode: {self.nan_mode}, "
            f"random_state: {self.random_seed})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict,
    catboost_train_dir_path: str) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_inputs (pd.DataFrame): The training data inputs.
        train_targets (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.
        catboost_train_dir_path (str): Path to the directory where the model

    Returns:
        'Classifier': The classifier model
    """
    hyperparameters['train_dir'] = catboost_train_dir_path
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
