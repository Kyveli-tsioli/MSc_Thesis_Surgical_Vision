ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=True,  #was False initially
    static_mlp=False

)
OptimizationParams = dict(
    dataloader=True,
    iterations = 30000, #default is 15000
    batch_size=1,
    coarse_iterations = 30000,  #default is 3000
    densify_until_iter = 15_000, #dfault is 10_000
    opacity_reset_interval = 3000,
    # opacity_reset_interval = 60000, #on 4dgs is commented out
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
    lambda_smoothness=0.9 # Added this for smoothness loss-24/06
)