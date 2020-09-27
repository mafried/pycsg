from pycsg.primitives import *

cube = {
    "n_primitives": 1,
    "box": True,
    "cylinder": False,
    "sphere": False,
    "operation": None
}

halfcylinder = {
    "n_primitives": 2,
    "box": True,
    "cylinder": True,
    "sphere": False,
    "operation": "difference"
}

halfsphere = {
    "n_primitives": 2,
    "box": True,
    "cylinder": False,
    "sphere": False,
    "operation": "difference"
}

patterns_dict = {"cube": cube, "halfcylinder": halfcylinder, "halfsphere": halfsphere}

