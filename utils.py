import numpy as np
import matplotlib.pyplot as plt
import gc
import torch

class MetricCollector:
    """Сборщик метрик обучения
    """
    def __init__(self, num_blocks, num_iter):
        """
        num_blocks: кол-во блоков, для которых ведется учет  
        """
        self.num_blocks = num_blocks
        self.num_iter = num_iter
        
        # списки для метрик  
        self.train_acc = [[None] * self.num_iter for _ in range(self.num_blocks)]
        self.train_err = [[None] * self.num_iter for _ in range(self.num_blocks)]
        self.val_acc = [[None] * self.num_iter for _ in range(self.num_blocks)]
        self.val_err = [[None] * self.num_iter for _ in range(self.num_blocks)]
        
        # Кумулятивные списки метрик
        self.train_acc_sum = [None] * self.num_blocks
        self.train_err_sum = [None] * self.num_blocks
        self.val_acc_sum = [None] * self.num_blocks
        self.val_err_sum = [None] * self.num_blocks
        
        # Метрики по батчам
        self.train_batch_acc = []
        self.train_batch_error = []
        self.val_batch_acc = [] 
        self.val_batch_error = []
    
    def summary_block(self, n):
        """
        train_acc, train_err, val_acc, val_err - порядок вывода
        """
        # считаем метрики
        self.train_acc_sum[n] = np.array(self.train_acc[n]).mean()
        self.train_err_sum[n] = np.array(self.train_err[n]).mean()
        self.val_acc_sum[n] = np.array(self.val_acc[n]).mean()
        self.val_err_sum[n] = np.array(self.val_err[n]).mean()
        
        return self.train_acc_sum[n], self.train_err_sum[n], self.val_acc_sum[n], self.val_err_sum[n]

def plot_summary(collector):
    """Выводит 4 графика:
    Accuracy = f(depth)
    Error = f(depth)
    Accuracy = f(num_batches)
    Error = f(num_batches)

    Args:
        collector MetricCollector: контейнер с метриками
    """
    plt.style.use('seaborn-whitegrid')
    _, ax = plt.subplots(3, 2, constrained_layout=True, figsize=(20, 10))
    ax[0][0].set_title('Loss')
    ax[0][0].set_xlabel('depth', loc='right')
    ax[0][0].set_xlim(left=1, right=collector.num_blocks)
    ax[0][1].set_title('Accuracy')
    ax[0][1].set_xlabel('depth', loc='right')
    ax[0][1].set_xlim(left=1, right=collector.num_blocks)
    ax[1][0].set_xlabel('num_batches', loc='right')
    ax[1][0].set_xlim(left=1, right=len(collector.train_batch_error))
    ax[1][1].set_xlabel('num_batches', loc='right')
    ax[1][1].set_xlim(left=1, right=len(collector.train_batch_acc))
    ax[2][0].set_xlabel('num_batches', loc='right')
    ax[2][0].set_xlim(left=1, right=len(collector.val_batch_error))
    ax[2][1].set_xlabel('num_batches', loc='right')
    ax[2][1].set_xlim(left=1,right=len(collector.val_batch_error))
    depth = [i+1 for i in range(collector.num_blocks)]
    n_b_train = [i+1 for i in range(len(collector.train_batch_error))]
    n_b_val = [i+1 for i in range(len(collector.val_batch_error))]
    # error_train = f(depth)
    ax[0][0].plot(depth, collector.train_err_sum, color='blue',lw = 3)
    # error_val = f(depth)
    ax[0][0].plot(depth, collector.val_err_sum, color='green', lw = 5)
    # acc_train = f(depth)
    ax[0][1].plot(depth, collector.train_acc_sum, color='blue', lw = 3)
    # acc_val = f(depth)
    ax[0][1].plot(depth, collector.val_acc_sum, color='green', lw = 5)
    # error_train = f(n_b)
    ax[1][0].plot(n_b_train, collector.train_batch_error, color='blue', lw = 3)
    # error_val = f(n_b)
    ax[2][0].plot(n_b_val, collector.val_batch_error, color='green', lw = 2)
    # acc_train = f(n_b)
    ax[1][1].plot(n_b_train, collector.train_batch_acc, color='blue', lw = 3)
    # acc_val = f(n_b)
    ax[2][1].plot(n_b_val, collector.val_batch_acc, color='green', lw = 2)
    plt.show()

# 2 функции для фиксации потребляемой  памяти
def mem_start():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def mem_end():
    torch.cuda.synchronize()
    print(torch.cuda.max_memory_allocated())