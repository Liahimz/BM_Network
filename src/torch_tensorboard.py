import torch.utils.tensorboard as tb

class Statistics:
    def __init__(self, generator_len, tb_writer: tb.writer.SummaryWriter = None, name=''):
        self.idx = []
        self.accuracy = []
        self.loss = []
        self.generator_len = generator_len
        self.tb_writer_ = tb_writer
        self.name = '/' + name if name else ''

    def append(self, loss, accuracy, global_step):
        self.idx.append(global_step / self.generator_len)
        self.accuracy.append(accuracy)
        self.loss.append(loss)

        if self.tb_writer_ is not None:
            self.tb_writer_.add_scalar(f'loss{self.name}', loss, global_step)
            self.tb_writer_.add_scalar(f'accuracy{self.name}', accuracy, global_step)