# “华为杯”第十八届中国研究生 数学建模竞赛
2021年中国研究生数学建模竞赛D题 - 抗乳腺癌候选药物的优化建模 (数据分析类)
## 题目介绍
问题1(特征筛选，清洗数据). 根据文件“Molecular_Descriptor.xlsx”和“ERα_activity.xlsx”提供的数据，针对1974个化合物的729个分子描述符进行变量选择，根据变量对生物活性影响的重要性进行排序，并给出前20个对生物活性最具有显著影响的分子描述符（即变量），并请详细说明分子描述符筛选过程及其合理性。

问题2(建立预测模型). 请结合问题1，选择不超过20个分子描述符变量，构建化合物对ERα生物活性的定量预测模型，请叙述建模过程。然后使用构建的预测模型，对文件“ERα_activity.xlsx”的test表中的50个化合物进行IC50值和对应的pIC50值预测，并将结果分别填入“ERα_activity.xlsx”的test表中的IC50_nM列及对应的pIC50列。

问题3(建立分类模型). 请利用文件“Molecular_Descriptor.xlsx”提供的729个分子描述符，针对文件“ADMET.xlsx”中提供的1974个化合物的ADMET数据，分别构建化合物的Caco-2、CYP3A4、hERG、HOB、MN的分类预测模型，并简要叙述建模过程。然后使用所构建的5个分类预测模型，对文件“ADMET.xlsx”的test表中的50个化合物进行相应的预测，并将结果填入“ADMET.xlsx”的test表中对应的Caco-2、CYP3A4、hERG、HOB、MN列。

问题4(建立优化模型). 寻找并阐述化合物的哪些分子描述符，以及这些分子描述符在什么取值或者处于什么取值范围时，能够使化合物对抑制ERα具有更好的生物活性，同时具有更好的ADMET性质（给定的五个ADMET性质中，至少三个性质较好）。

##  思路
由题目可知，D题的整体建模思路非常的清晰明了。只需要寻找相应问题的解决算法就好。对于各种算法比如特征筛选，分类，预测，优化都有非常多的算法，选择性能较好的5-10种即可。
当然各个问题中有需要注意的地方，以下列举一些比较关键的得分点。

问题一：

1.需要选出特定变量，不能得出没有具体含义的变量。

2.多种变量的相关性检验。

3.尽量选择5种左右特征选择算法，然后再进行考虑。

4.独立性。

问题二：

1.建模前的数据处理，第一问中变量独立性，相关性。

2.建模方法的选择和超参数的选择逻辑。

3.尽量选择5-10种预测模型。

4.模型的评价指标和分析。

5.模型性能好坏。

问题三：

1.建模前数据处理。

2.建模方法的选择和超参数的逻辑。

3.尽量选择5-10种预测模型。

4.模型的评价指标和分析。

5.模型性能好坏。

问题四：

1.对于优化问题一般可以使用遗传算法和粒子群算法这类搜索算法去搜。

2.尽量多使用几种优化算法。


## 文档说明

赛题文件：包含了官方给出的所有的比赛文件。

问题x代码：包含了问题x的所用代码。

graph：包含了论文中图的代码，包括各种漂亮的图。

论文：给了出往年数据分析的优秀论文。

## 致谢

这里感谢我的队友黄生辉和王宇琦一起奋斗3天拿下了比赛。

