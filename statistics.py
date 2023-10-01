import collections
import os
import re

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimSun'] # 指定使用宋体字体

# 车牌数据集
char_counter = collections.Counter()

# 正则表达式
zh_pattern = re.compile(u'[\u4e00-\u9fa5]')
num_pattern = re.compile('\d')

all_str = ""

for item in os.listdir(os.path.join('./cnn_datasets')):
    item=item[:-4]
    all_str = all_str + item

# 分别提取中文字符、数字和字母
zh_chars = zh_pattern.findall(all_str)
num_chars = num_pattern.findall(all_str)
en_chars = [c for c in all_str if c.isalpha() and c not in zh_chars]

# 统计出现次数
for c in zh_chars + num_chars + en_chars:
    char_counter[c] += 1

# 绘制柱状图
chars = list(char_counter.keys())
counts = list(char_counter.values())

# 隐藏y轴刻度标签
# plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

# 调整x轴标签之间的间距
plt.subplots_adjust(wspace=0.1)

plt.bar(chars, counts)
plt.xlabel('字符')
plt.ylabel('出现次数')
plt.title('车牌数据集字符出现次数统计')

# 调整图表大小
fig = plt.gcf()
fig.set_size_inches(12, 7)

plt.show()
