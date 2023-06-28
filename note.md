# R-BERT错题集

2023-06-28

> 本项目始于2023-06-21，囿于本人代码水平，历时约60小时才基本成功复现。
代码基本是一字一句打上去的，每一行代码都蕴含着心血。精力所限，现在实在无法再优化这屎代码了。
本文档记录这5天来的心路历程，并详解各代码部分，适当拓展相关知识点。


相比于本人之前的深度学习代码，本次项目的进步如下：

- 使用ipynb初步编程，并最后模块化拆分到各个.py文件
- 更改accuracy判断方式
- 增加log日志输出
- 优化各部分架构，引进checkpoint机制

**难点突破**：

- 在`TextDataset.__init__`中使用正则表达式提取目标文本
- 在model中使用`torch.mul()`对应元素相乘*（又称element-wise product、 element-wise multiplication或 Hadamard product）*，借助**boardcast机制**在batch中提取**entity**向量，避免使用循环（现在还没找到batch的map函数）

[TOC]

## 1. 数据集和模型的下载

[`main.py`](https://github.com/Silverwolf-x/r-bert/blob/master/code/main.py) --- `get_pretrain()`

**bert-base-uncased**模型通过huggingface的`transformers`包的`AutoTokenizer`, `AutoModel`下载。

需要注意的是，不能照抄huggingface样例的`transformers.AutoModelForMaskedLM`。它最后输出的是每个Token的Softmax得分，shape为`(batch_size, sequence_length, vocabulary_size)`。这里全量词表的大小`vocabulary_size`为30522(BERT-base)或50257(BERT-large)。而`transformers.AutoModel`最后输出BERT最后一层的隐藏状态，即shape为` (batch_size, sequence_length, hidden_size)`，这才是**R-BERT**中后续连接层需要的预训练模型。

> 下载**BERT**时，使用`mirror=’tuna’`清华镜像加速国内的模型下载。
>
> 下载**SemEval2010_task8_all_data**时，一开始调用datasets包无法直接连接github下载。最后挂外网，设置git端口后使用git clone下载。

## 2. 数据集处理

[`data_load.py`](https://github.com/Silverwolf-x/r-bert/blob/master/code/main.py)

数据集中的关键文件是`TRAIN_FILE.TXT`，`TEST_FILE_FULL.TXT`。他们的样式如下：

```text
8001	"The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
Message-Topic(e1,e2)
Comment: Assuming an audit = an audit document.

8002	"The <e1>company</e1> fabricates plastic <e2>chairs</e2>."
Product-Producer(e2,e1)
Comment: (a) is satisfied

```

输入方面，txt中每4行为一组数据。我们感兴趣的数据为整个句子，**entity**位置信息，关联label。

在`clean_data()`中，依据论文示例，用`$`和`#`替换句子中的`e1`和`e2`，且各占用一个单词位置；提取出各句子的关联label，并做数字编码`label_coder()`。

在`TextDataset`中，句子需要经过**BERT**模型的tokenizer拆分，以便后续传入**BERT**模型训练；在拆分后的含有[CLS]、[PAD]、[SEP]的句子中，确定entity位置信息。值得注意的是，通过遍历，发现max token length*（所有文本拆分后最大的拆分段数）*为125，小于论文中padding步骤的参数Max sentence length=128。因此不需要担心本数据集中有句子被切断。

输出方面，`TextDataset `返回`{'input_ids','token_type_ids','attention_mask'},label`。其中的字典直接就是**BERT**的输入，省心安逸。

***！！！转换tensor、提取信息等数据处理在`TextDataset`的初始化中`__init__`中进行，使得在初始化`TextDataset`时完成处理。不要在`__getitem__`中处理！这个问题的本质是尽量批处理，不要一个一个处理*。**

## 难点：确定tokenize后entity对应的词向量位置

[`data_load.py`](https://github.com/Silverwolf-x/r-bert/blob/master/code/main.py) --- `find_pos()`

根据单词的长度，是否在全量词表*(vocabulary)*中等因素，一个单词tokenize后可能会拆分为多个词向量。因此index极易错位。

思路1：先找到`$`的token编码是1002，寻找文本tokenize后前两个1002的位置。

> 不行，万一有$230这种美元符号在前面呢？

思路2：先找到`$ xxx $`部分的token编码为[1002,x,1002]，之后在文本tokenize后的纯数字列表中，寻找形如[1002,x,1002]的部分。

> 你说得对，但是怎么提取文本中`$ xxx $`部分呢？而且有些文本框住的不只一个单词。

综上所述，经过遍历发现，数据集中有如下几类特殊情形：

```text
The $ state $ has assembled a $230 million
$ x-rays $
$ log- jam $
```

**思路2实现：使用正则表达式**

好消息是，正则表达式可以满足上述要求，在文本中定位出`$ xxx $`部分，且能覆盖本数据集的所有情况。坏消息是，正则表达式晦涩难懂。但更好的消息是，我们有**ChatGPT**！

### 1） 正则表达式匹配目标文本

```python
re.search(r'\$ ([\w\-]+(?: [\w\-]+)*) \$', text).group()
```

> `([\w\-]+(?: [\w\-]+)*)` 表示匹配一个或多个用空格分隔的单词或连字符组成的字符串，并将其作为一个捕获组返回
>
> `\$` 匹配 $ 符号。
>
> `()`：匹配并捕获一组字符。
>
> `[\w\-]+`：匹配任意数量的单词字符或连字符。其中 `\w` 匹配字母、数字或下划线，`\-` 表示连字符。
>
> `(?: [\w\-]+)` 表示有字符 0 次或多次。其中`(?: )`表示非捕获括号，可以理解为option匹配，`*`表示匹配前面的字符 0 次或多次。

`re.search(xx,text)`返回与目标相匹配的信息，然后使用`.group()`提取得到的match文本。本数据集的每个文本有且只有一处match。

之后对match进行`tokenizer.encode()`数字编码*（如[1002,x,1002]）*，并使用循环，在文本编码中，寻找与其完全匹配的位置（有且只有一处），并记录始末index。

### 2）函数映射至所有文本

> 注意到，`TextDataset`中的输入是所有文本的tensor形式编码，shape为`(len_data,seq_length)`*（这里seq_length在tokenizer时被padding为统一的128）*。即，本问题可以抽象为：在shape为`(batch,x,y)`的张量中，**对每个batch中的向量/矩阵进行函数映射（结果shape不一定与输入相同）**。

本问题中，由于上述过程在`TextDataset.__init__`中完成，且处理过程仍在CPU上操作，因此采用python自带的`map()`处理效率不会损失太多。主要代码如下：

1. `x.unbind(dim=0)`：将shape为`(len_data,seq_length)`去掉len_data的维度变成一维tuple，其元素是一维tensor向量。`map()`要求的对象不能是numpy、tensor这类有shape的高维数据类型，而是tuple,list这类只有len的一维数据。
2. `torch.stack(x)`：默认`dim=0`，按照行（上下堆叠）tuple,list这类只有len的一维数据，返回tensor，shape为`(len(x),x.shape)`。

## 3. 模型搭建

[`model.py`](https://github.com/Silverwolf-x/r-bert/blob/master/code/model.py)

承接`TextDataset`输出中的字典部分，立刻得到**BERT**预训练模型的输出，其中有以下方法调用。

||shape|说明|备注|
| ---- | ---- | ---- | ---- |
| `last_hidden_state` | (batch_size, sequence_length, hidden_size) | **BERT**最后一层hidden_layer的输出 | hidden_size=768 |
| `pooler_output` | (batch_size, hidden_size) | CLS的hidden_layer→FC→Tanh()，通常用于句子分类 |      |
| `hidden_states` | (layer_counter, batch_size, sequence_length, hidden_size) | [0]是embedding，其余元素是各神经网络层的输出 | `bert(output_hidden_states=True)` |
| `attentions` |      | 各神经网络层的attention score | `output_attentions=True` |

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />
</p>
模型搭建Q&A

1. Entity1和Entity2如何共享一个layer

   `forword()`使用同一个`self.entity_fc`即可

2. dropout、tanh、fc的顺序

   >  $H_0^{'}= W_0 (\tanh(H_0)) + b_0$
   >
   > We add dropout before each add-on layer.

   依据原文可以肯定，先tanh后fc。dropout的作用是在更新神经网络参数时，输入端随机去掉值避免过拟合。参考现有R_BERT的代码，dropout放在tanh前。

3. $H_i-H_j$求均值时是否包含$\#的词向量

   > both the special separate tokens and the hidden entity vectors make important contributions to our approach

   依据原文，$\#的词向量是包含的。

4. $H_0$即CLS，用last_hidden_state还是pooler_output

   > For the final hidden state vector of the first token (i.e. ‘[CLS]’), we also add an activation operation and a fully connected layer

   依据原文，应该是用last_hidden_state。

   实践是检验真理的唯一标准，试验结果如下：

   - last_hidden_state：`logs\2023-06-27\2023-06-27-17.01.log`

   ```yaml
   [2023-06-27-17.10][trainer.py][line:154][INFO] Valid | Loss: 4.06e-03	 Acc: 0.8425	 F1 Score:81.34
   [2023-06-27-17.10][main.py][line:92][INFO] Score | Macro-averaged F1-scores (excluding Other): 81.3368
   ```

   - pooler_output：`logs\2023-06-27\2023-06-27-17.47.log`

   ```yaml
   [2023-06-27-17.56][trainer.py][line:154][INFO] Valid | Loss: 3.87e-03	 Acc: 0.8458	 F1 Score:80.98
   [2023-06-27-17.56][main.py][line:94][INFO] Score | Macro-averaged F1-scores (excluding Other): 80.9828
   ```

   似乎差别也不大，似乎pooler_output会更好？总而言之，依据论文，先用last_hidden_state吧。

5. Adam还是AdamW

   原文使用Adam，那么看起来更高级的AdamW呢？

   - Adam：上述两个例子都是Adam

   - AdamW：`logs\2023-06-27\2023-06-27-17.01.log`

   ```yaml
   [2023-06-27-17.10][trainer.py][line:154][INFO] Valid | Loss: 4.06e-03	 Acc: 0.8425	 F1 Score:81.34
   [2023-06-27-17.10][main.py][line:92][INFO] Score | Macro-averaged F1-scores (excluding Other): 81.3368
   ```
   看起来还不如Adam


> 令人费解的是，我弄了4天的训练效果都只有0.2的准确率，epoch100次才达到0.64。当我百思不得其解，重构了`data_load`.py,`score.py`,entity处理等多个步骤后，还是没有长进。最后检查到model部分，删去`with torch.no_grad()`，放开bert参数一起去训练，准确率就从0.78起步了。

***!!!使用预训练模型时，需要连同它的参数一起微调训练(fine-tune)***

## 难点：在batch中提取**entity**向量部分

[`model.py`](https://github.com/Silverwolf-x/r-bert/blob/master/code/model.py) --- entity_average()

前文提及，一个单词tokenize后的编码*（又称encoder、input_ids）*可能不唯一，而**BERT**输出的词向量（**word embedding**）个数与输入编码的个数一一对应。前文已经得到了目标词的词向量位置，现在的问题本质同前一个难点，都可以归结为：

**对每个batch中的向量/矩阵进行函数映射（结果shape不一定与输入相同）**

思路1：仿照前文，尝试map处理

> 注意到python原生map函数的诸多限制（前文已经提及），且这个任务在**R-BERT**模型中执行，需要考虑效率问题，不宜使用循环处理。

思路2：借用mask0-1向量，并使用tensor乘法提取出**entity**部分。

### 1）**mask vector**:将位置index转换为0-1向量

有两种方法：

- 借助for循环，为`torch,zeros()`指定区域赋值为1

- 借助**tensor>常数**的`True-False`判断，bool转化为float。这里float是为了后续与词向量（小数形式）相乘

  ```python
  print(torch.tensor([1, 2, 3, 4]) >= 2)
  # tensor([False,  True,  True,  True])
  ```

  ![mask_vector](https://github-production-user-asset-6210df.s3.amazonaws.com/104623550/249540560-60f68896-86f0-4681-b2b9-2264c37f471f.png)


  > tensor，numpy有这种方法，list，tuple没有。其中tensor的bool转换需用`x.float()`，numpy需用`x.astype(float)`。

### 2）**element-wise**：利用broadcast机制进行元素相乘

- **boardcast机制**

  在运行tensor乘法前，在第n个维度扩充为两者的最大值，以合乎数学规范，之后再进行对应元素相乘。

  >  注意，虽然有时允许`(3,)*(4,1)`，但这种乘法不稳定。推荐先`unsqueeze()`变成一样多的维度个数，再进行相乘 ，如`(3,1)*(4,1)`。这样broadcast机制就能稳定运行了。

  boardcast机制的代码本质是`x.expand(target_shape)`可以用以下代码验证：

  ```python
  a = torch.randn(3,4,1,4)
  b = torch.randn(1,4,2,1)
  if a.dim()!=b.dim():
      print("dims are not the same")
  else:
      target_shape = torch.zeros(a.dim())
      for i,(a_dim_size, b_dim_size) in enumerate(zip(a.shape,b.shape)):
          target_shape[i] = max(a_dim_size, b_dim_size)
      target_shape = target_shape.int().tolist()
      a_expand=a.expand(target_shape)
      b_expand=b.expand(target_shape)
      print(f'{a.shape=}\n{b.shape=}\n{target_shape=}')
      print("a*b == a_expand*b_expand ?")
      print((a*b == a_expand*b_expand).any())# True
  ```

- `tensor.mul(x,y)`<=>`x*y`：对应元素相乘，支持boardcast

  ```python
  # 数乘：常数&向量or矩阵
  x = torch.tensor([[1,2,3],[4,5,6]])# shape=(2,3)
  print(5*x)# tensor([[5,10,15],[20,25,30]])
  # 向量&向量
  a = torch.tensor([1,2,3])# （3,）
  b = torch.tensor([1,2,3,4]).view(-1,1)# (4，)-->（4,1）
  a*b #tensor([[1,2,3],[2,4,6],[3,6,9],[4,8,12]])
  # 本质是(3,)-->(1,3)，然后(1,3)*(4,1)进行broadcast为(4,3)
  # 这里(1,3)是行向量形式，通过行复制（上下复制）为(4,3)
  # 这里(4,1)是列向量形式，通过列复制（左右复制）为(4,3)
  ```

- `tensor.matmul(x,y)`<=>`x@y`：矩阵乘法，支持boardcast
  - `torch.dot(x,y)`：1D情况，不支持boardcast
  - `torch.mm(x,y)`：2D情况，不支持boardcast
  - `torch.bmm(x,y)`：3D情况，对每个 batch进行对应的矩阵乘法，不支持boardcast

本部分的示意图如下：

![element_wise](https://github-production-user-asset-6210df.s3.amazonaws.com/104623550/249542085-91e36493-08d2-42dc-aeaf-5bf8f4b1f757.png)

其中`mask.unsqueeze(-1)`的理解，即(2,3)-->(2,3,1)的理解：

(2,3)表示两行，每一行有3个数；(2,3,1)表示有2组数据，每组数据是(3,1)的列向量。某种意义上进行了自动转置。

## 4. 评分

[`scorer.py`](https://github.com/Silverwolf-x/r-bert/blob/master/code/scorer.py)

数据集的标准打分文件是2010年的perl文件，需要perl环境。本人试图把这个文件转换为主流的是python文件，但由于本人呢看不懂，难以喂给GPT，且其打分过程还有随机的skip。因此只能用`sklearn.metrics.f1_score`计算排除掉Other类的macro打分，以近似替代原结果。

## 随笔与致谢

这是现在我向深度学习领域踏出的第一步。无论未来我是否从事这个领域，希望未来当我遇到困难时，不要忘记这一段时间，痛苦且快乐的代码时光。与未来的自己共勉！

大二下期末周第一周周三考完寿险精算，到期末周第二周周二，致我逝去的青春（

很累，一个人看论文，代码思考，代码优化 ，调试，以及后期的画图，写作，上传到github，有很多的事情是第一次尝试，第一次摸索。很感谢有这样一个机会，让我近乎疯狂的与代码搏斗，还好随后险胜。写完代码，我深刻体会到码农的辛苦。老了，肝不动了。希望不要这么快头秃。以后也不会有精力去写这么详细的分析了，太累了，热情损耗的台快了，有点吃不消。

**致谢**

- 感谢MEDAI3给我机会复现这一篇论文，虽然很辛苦，但这又一次显著提升了我的代码能力。

- 感谢chatgpt的代码思路支持，虽然问题由代码编写转变为如何准确描述任务，也有不少难度。

- 感谢中国人民大学明德地下机房提供的RTX3080显卡支持。

- 感谢自己的认真、坚持和毅力，始终对这个项目不离不弃。

**小日记**

6.21与vscode的github连接斗智斗勇

6.22开始处理原数据txt转json，但后来放弃了，没必要应转换成字典来输入。不过这也启发我用字典传输变量，尤其是导入函数，方便快捷。

6.23完成所有code，8;00-23:00的含金量！

6.24更改了代码架构，从ipynb换为主流的project型分类

6.25平均跑一次需要一个半小时，第一次跑最后acc=0.3，F1为16。主要是学校CPU受限，numworker最高只能到3

6.26为什么不出结果呢？检查发现entity的提取过于粗暴，可能是数据处理的问题。问chatgpt，初步确定用正则表达式。下午要备考明天早上的毛概。

6.27完成正则化处理，但还是不行。最后下午发现是使用bert时`with torch.no_grad()`问题。完成md框架编写

6.28完成图示制作，完成md编写，完成github上传
