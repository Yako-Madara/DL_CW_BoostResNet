import torch.nn as nn
import torch.nn.functional as F

class ModelBuilder(nn.Module):
    """Класс, строящий текущую модель для обучения
    """
    def __init__(self, model, AllBlocks, n):
        super(ModelBuilder, self).__init__()
        self.model = model 
        self.AllBlocks = AllBlocks
        self.n = n 
        # Текущий обучаемый блок
        self.cur = self.AllBlocks[self.n]
        # Определим классификатор
        if self.n == 16:
            self.head = self.model.linear
        else:
            self.head = None   
        # Флаг создания полносвязного слоя
        self.flag = True 
    
    def forward(self, x):
        out = self.cur(x)
        # Если слой нулевой, то нам нужно только создать классификатор
        if self.n == 0:
            out = out.view(out.size(0), -1)
            in_feature = out.size(1)
            if self.flag:
                in_feature = out.size(1)    
                self.head = nn.Linear(in_feature, 10, device='cuda')
                self.flag = False
            out = self.head(out)
            return out
        # Если слой последний, то нам нужен классификатор от материнской модели
        elif self.n == 16:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.head(out)
            return out
        # Во всех остальных случаях будем использовать avg_pool2d и отдельный классификатор
        else:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            if self.flag:
                in_feature = out.size(1)
                self.head = nn.Linear(in_feature, 10, device='cuda')
                self.flag = False 
            out = self.head(out)
            return out   
        