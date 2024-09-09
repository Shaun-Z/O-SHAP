import importlib
import torch.utils.data
from explanations.base_explanation import BaseExplanation

def find_explanation_using_name(explanation_name):
    """Import the module "explanations/[explanation_name]_explanation.py".

    In the file, the class called ExplanationNameExplanation() will
    be instantiated. It has to be a subclass of BaseExplanation,
    and it is case-insensitive.
    """
    explanation_filename = "explanations." + explanation_name + "_explanation"
    explanationlib = importlib.import_module(explanation_filename)

    explanation = None
    target_explanation_name = explanation_name.replace('_', '') + 'explanation'
    for name, cls in explanationlib.__dict__.items():
        if name.lower() == target_explanation_name.lower() \
           and issubclass(cls, BaseExplanation):
            explanation = cls

    if explanation is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseExplanation with class name that matches %s in lowercase." % (explanation_filename, target_explanation_name))

    return explanation

def create_explanation(opt):
    """Create a explanation given the option.
    """
    explanation_class = find_explanation_using_name(opt.explanation_name)
    explanation = explanation_class(opt)
    return explanation