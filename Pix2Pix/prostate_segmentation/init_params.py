"""initialize parameters for constructing & training network"""


class InitParams:
    """Initializer"""
    dataset = 'NIH'
    run_id = 999994
    dir_in = r"F:\Minh\mxnet\projects\prostate_segmentation\inputs\NIH/"
    dir_out = "F:\BACKUPS\%s\outputs/run%d/" % (dataset, run_id)
    dir_retrieved_file = None
    retrieved_params = None
    log_file = 'log.txt'
    split = 1
    training_data_name = "im_normed"
    training_gt_name = "lab"
    im = None
    lab = None
    train_idx = None
    val_idx = None
    test_idx = None
    train_amount = 100
    val_amount = 50
    train_batch_size = 1
    val_batch_size = 1
    test_batch_size = 1
    train_iter = None
    val_iter = None
    test_iter = None
    gpu_id = 0
    ctx = None
    net = None
    to_review_network = False
    optimizer = 'adam'
    base_lr = 1e-3
    wd = .0000
    optimizer_params = None
    Trainer = None
    model = None
    prefix = "dmnet"
    save_interval = 1
    val_interval = 1
    steps = 10000
    epochs = 200
    log_interval = 20
    loss_term = 'DiceLoss'
    loss = None
    seed = 1
