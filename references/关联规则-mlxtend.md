# 关联规则与购物篮分析

返回[主入口](../SKILL.md)。本节回答同一笔交易内哪些商品共现，与数值变量相关系数不同。

## 先确认交易口径

至少有唯一交易键和商品/服务。客户ID不能直接当订单ID；跨店重号需构造唯一交易键。
写清期间、有效订单总数、退货/取消/赠品处理；缺交易键或商品名的记录单独报告，不能当未购买。
同一订单内同一商品购买多件计一次出现；保留有效单商品订单，不改变分母。
筛选商品后若产生空篮，明确是否仍在原分析总体内。月度渠道汇总不能还原购物篮。

## 计算示例

运行前检查现有Python、pandas与mlxtend版本，不假定宿主已安装。
以下使用已确认有效、无缺失的订单明细df，列为order_id与item；阈值只是示例，需按业务最小覆盖订单数调整。

```python
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

basket = pd.crosstab(df['order_id'], df['item']).gt(0)
n_transactions = len(basket)
if n_transactions == 0:
    raise ValueError('没有可分析的有效交易')
freq = fpgrowth(basket, min_support=0.02, use_colnames=True, max_len=3)
if freq.empty or not freq['itemsets'].map(len).ge(2).any():
    rules = pd.DataFrame()
else:
    rules = association_rules(
        freq, num_itemsets=n_transactions,
        metric='confidence', min_threshold=0.5,
    )
    rules['joint_orders'] = (rules['support'] * n_transactions).round().astype(int)
    # 正向搭售候选才筛lift>1；负关联分析不做这一过滤。
    rules = rules[rules['lift'].gt(1)].sort_values(
        ['joint_orders', 'lift'], ascending=False)
```

`num_itemsets`是原始交易数，不是频繁项集数。mlxtend 0.25.0普通无缺失用法中有默认值；示例显式传正确交易数。其他版本或缺失值模式需核对相应API。
FP-Growth可作常用入口，小数据可用Apriori交叉核对；计算开销取决于篮子数量、稀疏度、阈值和组合长度。
FP-Max仅输出极大频繁项集，可能缺少规则所需子集支持度，不直接替代freq计算完整规则。
不设通用“低于0.005会爆炸”阈值；无规则就如实说明，不无限降阈值凑结果。

## 解读与核验

| 指标 | 解读 |
|---|---|
| support | A、B共现订单数÷全部有效订单数 |
| confidence | 买A的订单中同时买B的比例，需与B总体购买率对照 |
| lift | confidence÷B总体购买率；大于1为样本内正关联，不等于显著或因果 |
| joint_orders | 实际覆盖订单数，小样本高lift不直接当稳定规律 |
| conviction | 无穷大可来自样本内confidence=1，不代表总体完全依赖 |

抽查至少一条规则的原始订单数，手算support、confidence和lift；重要候选看不同期间或分组是否稳定。
导出前将frozenset转换为排序文本，附频繁项集、规则、交易数、阈值、口径和处理记录。
结论写“样本共现事实／搭售解释待验证／下一步验证”，历史覆盖量不能写成新增销量。
并入已有三栏解释和报告，不再重复整套分析；缺依赖时保留口径与手算示例，不能伪称算法已运行。

## API依据

[mlxtend 0.25.0官方源码](https://github.com/rasbt/mlxtend/blob/v0.25.0/mlxtend/frequent_patterns/association_rules.py)。本示例依据该版本，运行时仍需核对安装版本。
