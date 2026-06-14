#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0  # geometry-only training: colors are frozen, no SH bands needed
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0  # geometry-only training: SH color features receive no gradient
        self.opacity_lr = 0.014
        self.lambda_dssim = 0.2  # unused while the RGB loss is disabled in train.py

        self.densification_interval = 500

        self.densify_from_iter = 500
        self.densify_until_iter = 25000

        self.random_background = False
        self.mask_threshold = 0.01
        self.lr_mask = 0.01
        
        self.nb_points = 3
        self.triangle_size = 2.23
        self.set_opacity = 0.28
        self.set_sigma =  1.16

        self.noise_lr = 5e5
        self.mask_dead = 0.08
        # Geometry-only loss weights, see notes/geometry_only_hyperparameters.md
        self.lambda_normals = 0.05   # rendered-normal vs depth-normal self-consistency (2DGS value)
        self.lambda_depth = 1.0      # primary supervision, takes the role of the old RGB loss
        self.depth_log_l1_weight = 0.8      # absolute (log-space) anchor; primary depth term
        self.depth_pearson_weight = 0.2     # scale/shift-invariant complementary regularizer
        self.depth_pearson_patch_size = 16  # 32
        self.depth_min_alpha = 0.1   # exclude pixels with accumulated opacity below this from the depth loss
        self.lambda_dist = 100.0     # depth distortion; 100 for unbounded scenes, 1000 for bounded
        self.lambda_opacity = 0.0055
        self.lambda_size = 0.00000001
        self.opacity_dead = 0.014
        self.importance_threshold = 0.022
        # Per-loss start iterations (replaces the old single iteration_mesh gate).
        # Tuned for the default 30k schedule; scale down for shorter runs
        # (e.g. 500 / 1500 for 5k iterations).
        self.depth_from_iter = 0
        self.dist_from_iter = 3000
        self.normal_from_iter = 7000

        self.cloning_sigma = 1.0
        self.cloning_opacity = 1.0
        self.lr_sigma = 0.0008
        self.lr_triangles_points_init = 0.0018

        self.proba_distr = 2 # 0 is based on opacity, 1 is based on sigma and 2 is alternating
        self.split_size = 24.0
        self.start_lr_sigma = 0
        self.max_noise_factor = 1.5


        # N triangle target and growth rate
        self.max_shapes = 1_000_000 # orig: 4M
        self.add_shape = 1.3 # growth per densification event; orig: 1.3

        # Error-aware densification: multiplicatively boosts a triangle's
        # sampling probability by (1 + beta * normalized_depth_error). 0 keeps
        # the original geometry-prior MCMC sampling; 2-5 is a sensible range.
        self.densify_error_beta = 0.0

        self.p = 1.6

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)