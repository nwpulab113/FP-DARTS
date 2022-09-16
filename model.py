import torch
import torch.nn as nn
from operationsB import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names1, indices1 = zip(*genotype.reduce1)
      op_names2, indices2 = zip(*genotype.reduce2)
      concat = genotype.reduce_concat
    else:
      op_names1, indices1 = zip(*genotype.normal1)
      op_names2, indices2 = zip(*genotype.normal2)
      concat = genotype.normal_concat
    self._compile(C, op_names1, indices1, op_names2, indices2, concat, reduction)

  def _compile(self, C, op_names1, indices1, op_names2, indices2, concat, reduction):
    assert len(op_names1) == len(indices1)
    assert len(op_names2) == len(indices2)
    self._steps = len(op_names1) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops1 = nn.ModuleList()
    for name, index in zip(op_names1, indices1):
      stride = 2 if reduction and index < 2 else 1
      op1 = OPS[name](C//2, stride, True)
      self._ops1 += [op1]
    self._indices1 = indices1

    self._ops2 = nn.ModuleList()
    for name, index in zip(op_names2, indices2):
      stride = 2 if reduction and index < 2 else 1
      op2 = OPS[name](C//2, stride, True)
      self._ops2 += [op2]
    self._indices2 = indices2
    self.p1 = nn.Conv2d(C , C//2, kernel_size=1)
    self.p2 = nn.Conv2d(C , C//2, kernel_size=1)


  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]#当前节点的前驱节点
    for i in range(self._steps):#0,1,2,3
      h1 = states[self._indices1[2*i]]#0,2,4,6  h1的第一个节点
      h2 = states[self._indices1[2*i+1]]#1,3,5,7  h2要连的第一个节点
      h3 = states[self._indices2[2 * i]]  # 0,2,4,6   h1的第二个节点
      h4 = states[self._indices2[2 * i + 1]]  # 1,3,5,7  h2要连的第二个节点

      op1 = self._ops1[2*i]
      op2 = self._ops1[2 * i + 1]
      op3 = self._ops2[2 * i]
      op4 = self._ops2[2*i+1]

      dim_1 = h1.shape[1]

      #p1=nn.Conv2d(dim_1, dim_1//2, kernel_size=1)
      h11=self.p1(h1)
      h12=self.p1(h3)
     # h11 = h1[:, :  dim_1 // 2, :, :]
     # h12 =h3[:, dim_1 // 2 : , :, :]
      h11 = op1(h11)
      h12 = op3(h12)
      h1=torch.cat((h11,h12),dim=1)


      #h1 = op1(h1)
      dim_2 = h2.shape[1]

      h21=self.p2(h2)
      h22=self.p2(h4)
 #     h21 = h2[:, :  dim_2 // 2, :, :]
#      h22 = h4[:, dim_2 // 2:, :, :]
      h21 = op2(h21)
      h22 = op4(h22)
      h2 = torch.cat((h21, h22), dim=1)
     # h2 = op2(h2)


      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity) :
          if not isinstance(op3,Identity):
              h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity) :
          if not isinstance(op4,Identity):
              h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C#当前模块的输出通道数
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

