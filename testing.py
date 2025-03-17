
import backend 
from torch import nn, Tensor
import torch
import numpy as np

def verify_node(node, expected_type, expected_shape, method_name):
    if expected_type == 'parameter':
        assert node is not None, (
            "{} should return an instance of nn.Parameter, not None".format(method_name))
        assert isinstance(node, nn.Parameter), (
            "{} should return an instance of nn.Parameter, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'loss':
        assert node is not None, (
            "{} should return an instance a loss node, not None".format(method_name))
        assert isinstance(node, (nn.modules.loss._Loss)), (
            "{} should return a loss node, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'tensor':
        assert node is not None, (
            "{} should return a node object, not None".format(method_name))
        assert isinstance(node, Tensor), (
            "{} should return a node object, instead got type {!r}".format(
            method_name, type(node).__name__))
    else:
        assert False, "If you see this message, please report a bug in the autograder"

    if expected_type != 'loss':
        assert all([(expected == '?' or actual == expected) for (actual, expected) in zip(node.detach().numpy().shape, expected_shape)]), (
            "{} should return an object with shape {}, got {}".format(
                method_name, expected_shape, node.shape))

def digit_class():
    import models
    model = models.DigitClassificationModel()
    dataset = backend.DigitClassificationDataset(model)

    model.train(dataset)
    va = dataset.get_validation_accuracy()
    print("VA: ", va)
digit_class()