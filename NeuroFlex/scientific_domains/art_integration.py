from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import KerasClassifier, PyTorchClassifier
from art.defences.preprocessor import Preprocessor
import numpy as np
import torch
import torch.nn as nn

class ARTPreprocessorWrapper(Preprocessor):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def __call__(self, x, y=None):
        return self.preprocessor(x)

    def fit(self, x, y=None, **kwargs):
        return None

    def estimate_gradient(self, x, grad):
        return grad

class ARTIntegration:
    def __init__(self, model=None, framework='keras'):
        self.model = model
        self.framework = framework
        self.art_classifier = None
        if model:
            self.set_model(model)

    def set_model(self, model, loss_fn=None, optimizer=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.art_classifier = self._create_art_classifier()

    def _create_art_classifier(self):
        if self.framework == 'keras':
            return KerasClassifier(model=self.model)
        elif self.framework == 'pytorch':
            loss = nn.CrossEntropyLoss()
            input_shape = tuple(self.model[0].in_features for _ in range(2))
            nb_classes = self.model[-2].out_features
            return PyTorchClassifier(model=self.model, loss=loss, input_shape=input_shape, nb_classes=nb_classes)
        else:
            raise ValueError("Unsupported framework. Use 'keras' or 'pytorch'.")

    def generate_adversarial_examples(self, x, method='fgsm', eps=0.3):
        if self.art_classifier is None:
            raise ValueError("Model not set. Call set_model() first.")
        if method == 'fgsm':
            attack = FastGradientMethod(estimator=self.art_classifier, eps=eps)
        elif method == 'pgd':
            attack = ProjectedGradientDescent(estimator=self.art_classifier, eps=eps)
        else:
            raise ValueError("Unsupported attack method. Use 'fgsm' or 'pgd'.")

        return attack.generate(x=x)

    def apply_defense(self, x, method='feature_squeezing', params=None):
        try:
            if method == 'feature_squeezing':
                from art.defences.preprocessor import FeatureSqueezing
                defense = FeatureSqueezing(bit_depth=params.get('bit_depth', 8))
            elif method == 'spatial_smoothing':
                from art.defences.preprocessor import SpatialSmoothing
                defense = SpatialSmoothing(window_size=params.get('window_size', 3))
            else:
                raise ValueError("Unsupported defense method. Use 'feature_squeezing' or 'spatial_smoothing'.")
            wrapped_defense = ARTPreprocessorWrapper(defense)
            return wrapped_defense(x)
        except ImportError:
            print(f"Warning: Defense method '{method}' could not be imported. Returning original input.")
            return x

    def adversarial_training(self, x, y, nb_epochs=3):
        if self.art_classifier is None:
            raise ValueError("Model not set. Call set_model() first.")
        from art.attacks.evasion import ProjectedGradientDescent
        pgd = ProjectedGradientDescent(self.art_classifier, eps=0.3, eps_step=0.1, max_iter=100)

        for _ in range(nb_epochs):
            adv_x = pgd.generate(x)
            self.art_classifier.fit(adv_x, y, nb_epochs=1)

    def evaluate_robustness(self, x, y, attack_methods=['fgsm', 'pgd']):
        if self.art_classifier is None:
            raise ValueError("Model not set. Call set_model() first.")
        results = {}
        for method in attack_methods:
            adv_x = self.generate_adversarial_examples(x, method=method)
            predictions = self.art_classifier.predict(adv_x)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
            results[method] = accuracy
        return results
