import logging
from torch.utils.tensorboard import SummaryWriter
import os


class Logger:

    def __init__(
            self,
            opt,
            logging_level=logging.INFO,
            file_level=logging.INFO,
            stream_level=logging.INFO
    ):
        self.opt = opt
        self.log_path = opt.log_path
        self.logging_level = logging_level

        self.file_level = file_level
        self.stream_level = stream_level

        self.logger = logging.getLogger('logger.log')
        self.logger.setLevel(self.logging_level)

        self.configure()
        
        # TensorBoard writer
        self.use_tb = opt.config.get('use_tensorboard', True)
        if self.use_tb:
            tb_dir = os.path.join(opt.experiments, 'tensorboard')
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_dir)
        else:
            self.tb_writer = None

    def configure(self):
        log_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.stream_level)
        stream_handler.setFormatter(log_format)

        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(log_format)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
    
    def log_scalar(self, tag, value, step):
        """记录标量到TensorBoard"""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, value_dict, step):
        """记录多个标量到TensorBoard"""
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(tag, value_dict, step)
    
    def close(self):
        """关闭TensorBoard writer"""
        if self.tb_writer is not None:
            self.tb_writer.close()
