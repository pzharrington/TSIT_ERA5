import torch
import importlib
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import *
from models.networks.encoder import *


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def create_network(cls, params):
    net = cls(params)
    net.print_network()
    net.init_weights(params.init_type, params.init_variance)
    return net


def define_G(params):
    netG_cls = find_network_using_name(params.netG, 'generator')
    return create_network(netG_cls, params)


def define_D(params):
    netD_cls = find_network_using_name(params.netD, 'discriminator')
    return create_network(netD_cls, params)


def define_E(params):
    # there exists only one encoder type
    netE_cls = find_network_using_name('conv', 'encoder')
    return create_network(netE_cls, params)
