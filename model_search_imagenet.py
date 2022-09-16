import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES,PRIMITIVES1,PRIMITIVES0
from genotypes import Genotype
import random
import numpy as np
import logging
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride,a):
    super(MixedOp, self).__init__()
    self._ops0 = nn.ModuleList()
    self._ops1 = nn.ModuleList()
    self.mp = nn.MaxPool2d(2, 2)
    self.k = 2
    self.a = a
    if self.a == [0]:
      for primitive in PRIMITIVES:
        op0 = OPS[primitive](C // self.k, stride, False)
        if 'pool' in primitive:
          op0 = nn.Sequential(op0, nn.BatchNorm2d(C // self.k, affine=False))
        self._ops0.append(op0)

    else:
      if self.a == [2]:
        for primitive in PRIMITIVES:
          op0 = OPS[primitive](C // self.k, stride, False)
          if 'pool' in primitive:
            op0 = nn.Sequential(op0, nn.BatchNorm2d(C // self.k, affine=False))
          self._ops0.append(op0)
        for primitive in PRIMITIVES1:
          op1 = OPS[primitive](C // self.k, stride, False)
          if 'pool' in primitive:
            op1 = nn.Sequential(op1, nn.BatchNorm2d(C // self.k, affine=False))
          self._ops1.append(op1)

      else:
        for primitive in PRIMITIVES1:
          op1 = OPS[primitive](C // self.k, stride, False)
          if 'pool' in primitive:
            op1 = nn.Sequential(op1, nn.BatchNorm2d(C // self.k, affine=False))
          self._ops1.append(op1)




  def forward(self, x, weights0, weights1):
    
     dim_2 = x.shape[1]
     xtemp = x[ : , :  dim_2//2, :, :]
     xtemp2 = x[ : ,  dim_2//2:, :, :]


     if self.a == [0]:
       temp10 = sum(w1.to(xtemp.device)* op1(xtemp) for w1, op1 in zip(weights0, self._ops0))  # 只对前1/4的特征进行计算
       if temp10.shape[2] == x.shape[2]:
         ans00 = torch.cat([temp10, xtemp2], dim=1)

       else:
         ans00 = torch.cat([temp10, self.mp(xtemp2)], dim=1)

       ans0 = channel_shuffle(ans00, self.k)
       ans = ans0

     else:
       if self.a == [2]:
         temp10 = sum(w1.to(xtemp.device) * op1(xtemp) for w1, op1 in zip(weights0, self._ops0))  # 只对前1/4的特征进行计算
         if temp10.shape[2] == x.shape[2]:
           ans00 = torch.cat([temp10, xtemp2], dim=1)
         else:
           ans00 = torch.cat([temp10, self.mp(xtemp2)], dim=1)
         ans0 = channel_shuffle(ans00, self.k)
         temp20 = sum(w2.to(xtemp2.device) * op2(xtemp2) for w2, op2 in zip(weights1, self._ops1))
         if temp20.shape[2] == x.shape[2]:
           ans11 = torch.cat([xtemp, temp20], dim=1)
    
         else: 
           ans11 = torch.cat([self.mp(xtemp), temp20], dim=1)
         ans1 = channel_shuffle(ans11, self.k)
         ans = ans1 + ans0

       else:
         temp20 = sum(w2.to(xtemp2.device) * op2(xtemp2) for w2, op2 in zip(weights1, self._ops1))

         if temp20.shape[2] == x.shape[2]:
           ans11 = torch.cat([xtemp, temp20], dim=1) 
         else:  
           ans11 = torch.cat([self.mp(xtemp), temp20], dim=1)
         ans1 = channel_shuffle(ans11, self.k)
         ans = ans1
     return ans
    

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        a = random.sample(range(0, 3), 1)
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride,a)
        self._ops.append(op)

  def forward(self, s0, s1,  weights0, weights1,weights2):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      #           print(222222222)
      s = sum(weights2[offset + j].to(self._ops[offset + j](h, weights0[offset + j], weights1[offset + j]).device) * self._ops[offset + j](h, weights0[offset + j], weights1[offset + j]) for j, h in
              enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights_inter = torch.cat([self.alphas_reduce1, self.alphas_reduce2], dim=1)
        weights = torch.sigmoid(weights_inter)

        weights0 = weights[:, : 4]
        weights1 = weights[:, 4:]
        n = 3
        start = 2
        weights2 = torch.sigmoid(self.betas_reduce[0:2])
        for i in range(self._steps - 1):
          end = start + n
          tw2 = torch.sigmoid(self.betas_reduce[start:end])
          start = end
          n += 1
          weights2 = torch.cat([weights2, tw2], dim=0)

      else:
        weights_inter = torch.cat([self.alphas_normal1, self.alphas_normal2], dim=1)
        weights = torch.sigmoid(weights_inter)
        weights0 = weights[:, : 4]
        weights1 = weights[:, 4:]
        n = 3
        start = 2
        weights2 = torch.sigmoid(self.betas_normal[0:2])
        for i in range(self._steps - 1):
          end = start + n
          tw2 = torch.sigmoid(self.betas_normal[start:end])
          start = end
          n += 1
          weights2 = torch.cat([weights2, tw2], dim=0)

      s0, s1 = s1, cell(s0, s1, weights0, weights1, weights2)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)
    self.alphas_normal1 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce1 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_normal2 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce2 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
    self.betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal1,
      self.alphas_reduce1,
      self.alphas_normal2,
      self.alphas_reduce2,
      self.betas_normal,
      self.betas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    def _parse(weights0, weights1, weights2):
      gene = []
      gene1 = []
      gene2 = []
      gene3 = []
      gene4 = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W0 = weights0[start:end].copy()
        W1 = weights1[start:end].copy()
        W = np.hstack((W0, W1))


        W2 = weights2[start:end].copy()

        for j in range(n):
          W[j, :] = W[j, :] * W2[j]


        edges0 = sorted(range(i + 2),
                        key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES0.index('none')))[:2]
        for j in edges0:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES0.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES0[k_best], j))
        start = end
        n += 1
      return gene

    n = 3
    start = 2
    weightsr2 = torch.sigmoid(self.betas_reduce[0:2])
    weightsn2 = torch.sigmoid(self.betas_normal[0:2])
    for i in range(self._steps - 1):
      end = start + n
      tw2 = torch.sigmoid(self.betas_reduce[start:end])
      tn2 = torch.sigmoid(self.betas_normal[start:end])
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2, tw2], dim=0)
      weightsn2 = torch.cat([weightsn2, tn2], dim=0)

    w_ir = torch.sigmoid(torch.cat([self.alphas_reduce1, self.alphas_reduce2], dim=1))
    w_ir1 = w_ir[:, : 4]
    w_ir2 = w_ir[:, 4:]

    w_in = torch.sigmoid(torch.cat([self.alphas_normal1, self.alphas_normal2], dim=1))
    w_in1 = w_in[:, : 4]
    w_in2 = w_in[:, 4:]

    gene_normal = _parse(w_in1.data.cpu().numpy(), w_in2.data.cpu().numpy(), weightsn2.data.cpu().numpy())
    gene_reduce = _parse(w_ir1.data.cpu().numpy(), w_ir2.data.cpu().numpy(), weightsr2.data.cpu().numpy())
    concat = range(2 + self._steps - self._multiplier, self._steps + 2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
