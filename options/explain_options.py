from .base_options import BaseOptions


class ExplainOptions(BaseOptions):
    """This class includes explain options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # default values
        parser.add_argument('--index_explain', nargs='+', default=[], help='the indices of the class to explain')
        parser.add_argument('--index_instance', type=int, nargs='+', default=[], help='the indices of the instance to explain')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during explanation.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='res_class')
        self.isTrain = False
        
        return parser