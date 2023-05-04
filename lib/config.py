from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = None
        self.lr = 1e-5
        self.enc_layer = 1
        self.dec_layer = 3
        self.nepoch = 10
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)
        
        if self.mem_feat_lambda is not None:
            self.mem_feat_lambda = float(self.mem_feat_lambda)
        
        
        if self.rel_mem_compute == 'None' :
            self.rel_mem_compute = None
        if self.obj_loss_weighting == 'None':
            self.obj_loss_weighting = None
        if self.rel_loss_weighting == 'None':
            self.rel_loss_weighting = None

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-save_path', default=None, type=str)
        parser.add_argument('-model_path', default=None, type=str)
        parser.add_argument('-data_path', default='data/ag/', type=str)
        parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=10, type=int)
        parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)

        #logging arguments
        parser.add_argument('-log_iter', default=100, type=int)
        parser.add_argument('-no_logging', action='store_true')

 
        # heads arguments
        parser.add_argument('-obj_head', default='gmm', type=str, help='classification head type')
        parser.add_argument('-rel_head', default='gmm', type=str, help='classification head type')
        parser.add_argument('-K', default=4, type=int, help='number of mixture models')

        # tracking arguments

        parser.add_argument('-tracking', action='store_true')
        # memory arguments
        parser.add_argument('-rel_mem_compute', default=None, type=str, help='compute relation memory hallucination [seperate/joint/None]')
        parser.add_argument('-obj_mem_compute', action='store_true')
        parser.add_argument('-take_obj_mem_feat', action='store_true')
        parser.add_argument('-obj_mem_weight_type',default='simple', type=str, help='type of memory [both/al/ep/simple]')
        parser.add_argument('-rel_mem_weight_type',default='simple', type=str, help='type of memory [both/al/ep/simple]')
        parser.add_argument('-mem_fusion',default='early', type=str, help='early/late')
        parser.add_argument('-mem_feat_selection',default='manual', type=str, help='manual/automated')
        parser.add_argument('-mem_feat_lambda',default=None, type=str, help='selection lambda')
        parser.add_argument('-pseudo_thresh', default=7, type=int, help='pseudo label threshold')

        # uncertainty arguments
        parser.add_argument('-obj_unc', action='store_true')
        parser.add_argument('-rel_unc', action='store_true')

        #loss arguments
        parser.add_argument('-obj_loss_weighting',default=None, type=str, help='ep/al/None')
        parser.add_argument('-rel_loss_weighting',default=None, type=str, help='ep/al/None')
        parser.add_argument('-mlm', action='store_true')
        parser.add_argument('-eos_coef',default=1,type=float,help='background class scaling in ce or nll loss')
        parser.add_argument('-obj_con_loss', default=None, type=str,  help='intra video visual consistency loss for objects (euc_con/info_nce)')
        parser.add_argument('-lambda_con', default=1,type=float,help='visual consistency loss coef')
        return parser
