import torch.utils.tensorboard as tb

class Statistics:
    def __init__(self, generator_len, tb_writer: tb.writer.SummaryWriter = None, name=''):
        self.idx = []
        self.accuracy = []
        self.loss = []
        self.val_loss = []
        self.val_acc = []
        self.generator_len = generator_len
        self.tb_writer_ = tb_writer
        self.name = '/' + name if name else ''

    def append(self, loss, accuracy, val_loss, val_acc, global_step):
        self.idx.append(global_step / self.generator_len)
        self.accuracy.append(accuracy)
        self.loss.append(loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

        if self.tb_writer_ is not None:
            self.tb_writer_.add_scalar(f'loss{self.name}', loss, global_step)
            self.tb_writer_.add_scalar(f'accuracy{self.name}', accuracy, global_step)
            self.tb_writer_.add_scalar(f'val_loss{self.name}', val_loss, global_step)
            self.tb_writer_.add_scalar(f'val_acc{self.name}', val_acc, global_step)