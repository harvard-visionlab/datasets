import torch, torchvision
import .datasets

dependencies = ['torch', 'torchvision']

def hello():
  def say_hello(name="World"):
    print(f"Hello {name}")

  return say_hello

def imagenette():
  return _datasets.imagenette
