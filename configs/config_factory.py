from .urban3d_baseline_rgb import *

from .urban3d_easy_rgb import *
from .urban3d_easy_d import *
from .urban3d_easy_rgbd import *
from .urban3d_easy_d2g import *

from .urban3d_optimal_rgbdt import *
from .urban3d_optimal_rgbd import *
from .urban3d_optimal_rgb import *
from .urban3d_optimal_d2g import *
from .urban3d_optimal_d import *

from .urban3d_experiment import *
from .urban3d_experimental_rgbd import *
from .urban3d_experimental_rgbdt_rpn import *

def create_config(name = "easy_rgb", inference = False):
    config = None

    if name == "baseline_rgb":
        config = Urban3dBaselineRGB() if not inference else Urban3dBaselineRGBInference()

    elif name == "optimal_rgb":
        config = Urban3dOptimalRGB() if not inference else Urban3dOptimalRGBInference()
    elif name == "optimal_d":
        config = Urban3dOptimalD() if not inference else Urban3dOptimalDInference()
    elif name == "optimal_d2g":
        config = Urban3dOptimalD2G() if not inference else Urban3dOptimalD2GInference()
    elif name == "optimal_rgbd":
        config = Urban3dOptimalRGBD() if not inference else Urban3dOptimalRGBDInference()
    elif name == "optimal_rgbdt":
        config = Urban3dOptimalRGBDT() if not inference else Urban3dOptimalRGBDTInference()

    elif name == "easy_rgb":
        config = Urban3dEasyRGB() if not inference else Urban3dEasyRGBInference()
    elif name == "easy_d":
        config = Urban3dEasyD() if not inference else Urban3dEasyDInference()
    elif name == "easy_d2g":
        config = Urban3dEasyD2G() if not inference else Urban3dEasyD2GInference()
    elif name == "easy_rgbd":
        config = Urban3dEasyRGBD() if not inference else Urban3dEasyRGBDInference()

    elif name == "experiment":
        config = Urban3dExperiment() if not inference else Urban3dExperimentInference()
    elif name == "experimental_rgbd":
        config = Urban3dExperimentalRGBD() if not inference else Urban3dExperimentalRGBDInference()
    elif name == "experimental_rgbdt_rpn":
        config = Urban3dExperimentalRGBDTRPN() if not inference else Urban3dExperimentalRGBDTRPNInference()

    # to be continued ...

    if config is None:
        raise RuntimeError(f"Config type {name} could not be found.")

    return config
