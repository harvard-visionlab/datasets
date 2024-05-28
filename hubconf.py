# import os
# import sys
# import torch, torchvision
# from contextlib import contextmanager

# @contextmanager
# def temporary_sys_path(path):
#     """Temporarily add a directory to sys.path."""
#     sys.path.insert(0, path)
#     yield
#     sys.path.pop(0)

# current_dir = os.path.dirname(os.path.abspath(__file__))

# dependencies = ['torch', 'torchvision']

# def hello():
#     def say_hello(name="World"):
#         print(f"Hello {name}")

#     return say_hello

# def imagenette(*args, **kwargs):
#     with temporary_sys_path(current_dir):
#         from datasets.images.imagenette import imagenette
#     return imagenette(*args, **kwargs)

# # ==============================================================
# #  some handy lower-level utils
# # ==============================================================

# def calculate_sha256(*args, **kwargs):
#     with temporary_sys_path(current_dir):
#         from datasets.utils import calculate_sha256
#     return calculate_sha256