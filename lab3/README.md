# Lab3 背包优化问题
## 分治法求解0-1背包问题代码详解

这段代码使用分治法（Divide and Conquer）来解决经典的0-1背包问题。我将从以下几个方面详细讲解：

1. 问题概述

0-1背包问题描述：给定一组物品，每个物品有重量和价值，在不超过背包承重的情况下，如何选择物品使总价值最大。

2. 方法概述

分治法将问题分解为更小的子问题，递归求解，然后合并结果。对于背包问题，每个物品都有选或不选两种选择。

3. 代码结构

3.1 类方法定义

```python
def divide_and_conquer(self) -> Tuple[int, float, List[int]]:
    """分治法求解0-1背包问题
    Returns:
        tuple: (最优值, 计算时间, 选择的物品列表)
    """
```

• 返回一个元组，包含最优值、计算时间和选择的物品列表

• 是类方法，可以访问类实例的属性和方法


3.2 初始化

```python
start_time = time.perf_counter()
self.selected_items = [0] * self.n
self.best_value = 0
self.best_selection = [0] * self.n
```

• `start_time`: 记录开始时间用于计算总耗时

• `selected_items`: 临时存储当前选择的物品状态

• `best_value`: 记录当前找到的最大价值

• `best_selection`: 记录当前最优解的选择方案


3.3 核心递归函数

```python
def solve(i: int, c: int, current_selection: List[int]) -> int:
```

• `i`: 当前考虑的物品索引（从n到0）

• `c`: 剩余的背包容量

• `current_selection`: 当前的选择方案列表


终止条件

```python
if time.perf_counter() - start_time > self.max_time:
    return -1
if i == 0 or c == 0:
    current_value = sum(v * s for v, s in zip(self.values, current_selection))
    if current_value > self.best_value:
        self.best_value = current_value
        self.best_selection = current_selection.copy()
    return 0
```

1. 超时检查：如果超过最大允许时间，返回-1
2. 递归终止条件：当没有物品可选(i=0)或背包容量为0(c=0)时
   • 计算当前选择方案的总价值

   • 如果比已知最优解更好，更新最优解

   • 返回0（因为此时不能再增加价值）


递归处理

```python
# 不选择当前物品
value1 = solve(i - 1, c, current_selection)

# 选择当前物品
if self.weights[i - 1] <= c:
    current_selection[i - 1] = 1
    value2 = self.values[i - 1] + solve(i - 1, c - self.weights[i - 1], current_selection)
    current_selection[i - 1] = 0
    return max(value1, value2)
return value1
```

1. 不选择当前物品：直接递归处理剩下的物品(i-1)，容量不变
2. 选择当前物品：
   • 检查当前物品是否能放入背包（重量≤剩余容量）

   • 如果能，标记选择该物品，递归处理剩下的物品和减少的容量

   • 回溯：取消选择标记

   • 返回选或不选中的较大值

3. 如果不能选择当前物品，直接返回不选的值

3.4 启动递归和结果处理

```python
current_selection = [0] * self.n
result = solve(self.n, self.capacity, current_selection)
end_time = time.perf_counter()

# 将选择的物品编号转换为索引列表
selected_items = [i for i, selected in enumerate(self.best_selection) if selected == 1]
return result, end_time - start_time, selected_items
```

1. 初始化选择列表全为0（未选择）
2. 从所有物品(n)和完整容量开始递归
3. 记录结束时间
4. 将最优解的选择方案转换为物品索引列表
5. 返回结果元组

4. 算法特点

• 时间复杂度：O(2^n)，因为每个物品都有选或不选两种选择

• 空间复杂度：O(n)用于递归栈和选择列表

• 优点：思路直观，易于理解

• 缺点：对于大规模问题效率低，没有利用重叠子问题的特性（动态规划更高效）


5. 改进方向

1. 记忆化：存储已计算的子问题结果，避免重复计算
2. 剪枝：提前终止不可能优于当前最优解的路径
3. 转换为动态规划：自底向上计算，更高效

这个实现虽然使用了分治法的思想，但实际上是穷举所有可能的组合，因此对于大规模问题性能较差。在实际应用中，通常会使用动态规划或其他优化方法来解决背包问题。

## 动态规划法求解0-1背包问题代码详解

这段代码使用动态规划(Dynamic Programming)来解决经典的0-1背包问题。动态规划是解决背包问题最有效的方法之一，相比分治法有更好的时间复杂度。

1. 动态规划思想概述

动态规划通过将问题分解为相互重叠的子问题，并存储子问题的解来避免重复计算。对于0-1背包问题，动态规划的核心思想是：

• 构建一个二维表格`dp`，其中`dp[i][w]`表示考虑前i个物品，在背包容量为w时能获得的最大价值

• 通过填表的方式逐步构建解

• 最后通过回溯确定具体选择了哪些物品


2. 代码结构详解

2.1 方法定义和初始化

```python
def dynamic_programming(self) -> Tuple[int, float, List[int]]:
    """动态规划求解0-1背包问题"""
    start_time = time.perf_counter()
```

• 返回元组：(最大价值, 计算时间, 选择的物品索引列表)

• `start_time`记录开始时间用于计算耗时


2.2 DP表和选择记录表初始化

```python
dp = [[0] * (self.capacity + 1) for _ in range(self.n + 1)]
selected = [[False] * (self.capacity + 1) for _ in range(self.n + 1)]
```

• `dp`表：`(n+1) x (capacity+1)`的二维数组，初始化为0

  • 行表示物品(0-n)，列表示容量(0-capacity)

• `selected`表：与`dp`表同样大小，记录是否选择了当前物品


2.3 填表过程

```python
for i in range(1, self.n + 1):
    for w in range(self.capacity + 1):
        if time.perf_counter() - start_time > self.max_time:
            return -1, time.perf_counter() - start_time, []
            
        if self.weights[i-1] <= w:
            if self.values[i-1] + dp[i-1][w-self.weights[i-1]] > dp[i-1][w]:
                dp[i][w] = self.values[i-1] + dp[i-1][w-self.weights[i-1]]
                selected[i][w] = True
            else:
                dp[i][w] = dp[i-1][w]
        else:
            dp[i][w] = dp[i-1][w]
```

1. 外层循环：遍历每个物品(i从1到n)
2. 内层循环：遍历每个可能的容量(w从0到capacity)
3. 超时检查：如果超过最大允许时间，返回当前结果
4. 物品选择逻辑：
   • 如果当前物品重量≤当前容量(`weights[i-1] <= w`)：

     ◦ 比较"选择当前物品"和"不选择"的价值

     ◦ 如果选择更优，更新`dp`表并标记`selected`表

   • 否则：直接继承不选择的值


2.4 回溯确定选择的物品

```python
selected_items = []
w = self.capacity
for i in range(self.n, 0, -1):
    if selected[i][w]:
        selected_items.append(i-1)
        w -= self.weights[i-1]
```

1. 从最后一个物品开始向前回溯
2. 如果`selected[i][w]`为True，表示选择了该物品：
   • 记录物品索引(转换为0-based)

   • 减少剩余容量

3. 继续检查前一个物品

2.5 返回结果

```python
return dp[self.n][self.capacity], time.perf_counter() - start_time, selected_items
```

返回元组：(最大价值, 计算时间, 选择的物品索引列表)

3. 算法分析

3.1 时间复杂度

• 填表过程：O(n×capacity)，双重循环

• 回溯过程：O(n)，单次遍历

• 总时间复杂度：O(n×capacity)


3.2 空间复杂度

• DP表：O(n×capacity)

• 选择记录表：O(n×capacity)

• 总空间复杂度：O(n×capacity)


3.3 优点

1. 避免了分治法的重复计算
2. 时间复杂度从指数级(O(2^n))降低到多项式级(O(n×capacity))
3. 可以准确找到最优解

3.4 局限性

1. 当capacity很大时，空间消耗可能成为问题
2. 伪多项式时间复杂度，不是严格的多项式时间算法

4. 示例说明

假设有以下输入：
• 容量capacity=5

• 物品：

  • 物品0：重量=2，价值=3

  • 物品1：重量=3，价值=4

  • 物品2：重量=4，价值=5


DP表填表过程：

| i\w | 0 | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|---|
| 0   | 0 | 0 | 0 | 0 | 0 | 0 |
| 1   | 0 | 0 | 3 | 3 | 3 | 3 |
| 2   | 0 | 0 | 3 | 4 | 4 | 7 |
| 3   | 0 | 0 | 3 | 4 | 5 | 7 |

回溯过程：

从dp[3][5]=7开始：
1. selected[3][5]=True → 选择物品2(重量4)
   • w = 5-4 = 1

2. selected[2][1]=False → 不选择物品1
3. selected[1][1]=False → 不选择物品0

最终选择物品2，总价值7

5. 优化方向

1. 空间优化：使用滚动数组将空间复杂度降为O(capacity)
2. 分支限界法：对于大规模问题，可以结合贪心算法进行优化
3. 记忆化搜索：自顶向下的实现方式，可能在某些情况下更直观

这段代码完整地实现了动态规划解决0-1背包问题的经典方法，是理解和学习动态规划思想的优秀示例。

## 贪心算法求解0-1背包问题代码详解

这段代码使用贪心算法(Greedy Algorithm)来解决0-1背包问题。贪心算法虽然不能保证得到最优解，但在某些情况下可以提供较好的近似解，且时间复杂度较低。

1. 贪心算法思想概述

贪心算法的核心思想是：在每一步选择中都采取当前状态下最好或最优（即最有利）的选择，从而希望导致结果是最好或最优的算法。对于0-1背包问题，常见的贪心策略有：

1. 按价值贪心：优先选择价值高的物品
2. 按重量贪心：优先选择重量轻的物品
3. 按性价比贪心：优先选择价值/重量比(性价比)高的物品

这段代码采用的是第三种策略——按性价比贪心。

2. 代码结构详解

2.1 方法定义和初始化

```python
def greedy(self) -> Tuple[int, float, List[int]]:
    """贪心算法求解0-1背包问题"""
    start_time = time.perf_counter()
```

• 返回元组：(获得的总价值, 计算时间, 选择的物品索引列表)

• `start_time`记录开始时间用于计算耗时


2.2 计算性价比并排序

```python
# 计算性价比并保存原始索引
items = [(i, self.values[i]/self.weights[i], self.weights[i], self.values[i]) 
        for i in range(self.n) if self.weights[i] > 0]
# 按性价比降序排序
items.sort(key=lambda x: x[1], reverse=True)
```

1. 创建一个包含4个元素的元组列表：
   • 原始索引(i)

   • 性价比(values[i]/weights[i])

   • 重量(weights[i])

   • 价值(values[i])

2. 过滤掉重量为0的物品(避免除以0错误)
3. 按性价比(第2个元素)降序排序

2.3 贪心选择过程

```python
total_value = 0
remaining_capacity = self.capacity
selected_items = []

for idx, _, weight, value in items:
    if time.perf_counter() - start_time > self.max_time:
        return -1, time.perf_counter() - start_time, []
    if weight <= remaining_capacity:
        selected_items.append(idx)
        total_value += value
        remaining_capacity -= weight
```

1. 初始化总价值、剩余容量和选择的物品列表
2. 遍历排序后的物品：
   • 检查是否超时

   • 如果当前物品可以放入背包(weight ≤ remaining_capacity)：

     ◦ 记录物品原始索引

     ◦ 增加总价值

     ◦ 减少剩余容量


2.4 返回结果

```python
return total_value, time.perf_counter() - start_time, sorted(selected_items)
```

返回元组：(获得的总价值, 计算时间, 按原始索引排序的物品列表)

3. 算法分析

3.1 时间复杂度

1. 计算性价比：O(n)，遍历所有物品
2. 排序：O(n log n)，使用Python的Timsort算法
3. 贪心选择：O(n)，遍历排序后的物品
4. 排序结果：O(k log k)，k是选择的物品数量
5. 总时间复杂度：O(n log n)

3.2 空间复杂度

1. 存储物品信息：O(n)
2. 选择结果存储：O(n)最坏情况
3. 总空间复杂度：O(n)

3.3 优点

1. 实现简单，易于理解
2. 时间复杂度低，适合大规模问题
3. 在某些情况下可以得到较好的近似解

3.4 局限性

1. 不能保证最优解：贪心算法对于0-1背包问题不能保证得到最优解
   • 示例：capacity=10, items=[(6,6), (5,5), (5,5)]

   • 贪心会选择(6,6)，但最优解是两个(5,5)

2. 依赖排序策略：不同的贪心策略可能得到不同的结果

4. 示例说明

假设有以下输入：
• 容量capacity=10

• 物品：

  • 物品0：重量=2，价值=6

  • 物品1：重量=5，价值=9

  • 物品2：重量=3，价值=5

  • 物品3：重量=4，价值=10


计算性价比：
• 物品0：6/2=3.0

• 物品1：9/5=1.8

• 物品2：5/3≈1.67

• 物品3：10/4=2.5


排序后顺序：
物品0(3.0) → 物品3(2.5) → 物品1(1.8) → 物品2(1.67)

贪心选择过程：
1. 选择物品0：重量2，价值6
   • 剩余容量=10-2=8

2. 选择物品3：重量4，价值10
   • 剩余容量=8-4=4

3. 无法选择物品1(重量5>4)
4. 选择物品2：重量3，价值5
   • 剩余容量=4-3=1


最终结果：
选择物品0、2、3，总价值=6+5+10=21

5. 优化与变种

1. 多种贪心策略比较：
   • 可以同时实现多种贪心策略(按价值、按重量、按性价比)

   • 最后选择结果最好的一个


2. 贪心+回溯：
   • 先用贪心算法得到一个解

   • 然后尝试用回溯法改进


3. 部分背包问题：
   • 如果可以分割物品，贪心算法可以得到最优解

   • 这种情况下可以按性价比排序后尽可能多地选择


这段代码展示了贪心算法在0-1背包问题中的典型应用，虽然不能保证最优解，但在很多实际应用中，由于其高效性，仍然是一个实用的选择。

## 分支限界法求解0-1背包问题代码详解

这段代码使用分支限界法(Branch and Bound)来解决0-1背包问题。分支限界法是一种智能化的穷举搜索算法，通过系统地搜索解空间并结合剪枝策略来提高效率。

1. 分支限界法思想概述

分支限界法的核心思想是：
• 分支：将问题分解为更小的子问题（节点）

• 限界：计算每个节点的价值上界（bound）

• 剪枝：如果节点的上界不可能优于当前最优解，则剪枝（不再探索该分支）

• 优先队列：优先探索最有希望的节点（上界最大的节点）


2. 代码结构详解

2.1 方法定义和初始化

```python
def branch_and_bound(self) -> Tuple[int, float, List[int]]:
    """分支限界法求解0-1背包问题"""
    start_time = time.perf_counter()
```

• 返回元组：(最大价值, 计算时间, 选择的物品索引列表)

• `start_time`记录开始时间用于计算耗时


2.2 节点类定义

```python
class Node:
    def __init__(self, level, value, weight, bound, selected):
        self.level = level  # 当前决策层级
        self.value = value  # 当前总价值
        self.weight = weight  # 当前总重量
        self.bound = bound  # 价值上界
        self.selected = selected.copy()  # 记录选择的物品
```

节点类存储搜索状态的关键信息：
• `level`：当前决策的物品层级

• `value`：当前路径的总价值

• `weight`：当前路径的总重量

• `bound`：该节点的价值上界

• `selected`：已选择的物品索引列表


2.3 上界计算函数

```python
def calculate_bound(node: Node) -> float:
    if node.weight >= self.capacity:
        return 0
    
    value_bound = node.value
    weight = node.weight
    level = node.level + 1
    
    while level < self.n and weight + self.weights[level] <= self.capacity:
        value_bound += self.values[level]
        weight += self.weights[level]
        level += 1
    
    if level < self.n:
        value_bound += (self.capacity - weight) * (self.values[level] / self.weights[level])
    
    return value_bound
```

计算节点价值上界的步骤：
1. 如果已超重，返回0（不可行）
2. 初始上界 = 当前价值
3. 尽可能多地装入后续物品（贪心）
4. 如果还有剩余容量，装入部分物品（分数背包思想）
5. 返回计算的上界值

2.4 物品排序预处理

```python
# 按照单位重量价值排序
items = [(i, self.values[i]/self.weights[i]) for i in range(self.n)]
items.sort(key=lambda x: x[1], reverse=True)
sorted_indices = [x[0] for x in items]

# 重新排序weights和values
original_weights = self.weights.copy()
original_values = self.values.copy()
self.weights = [self.weights[i] for i in sorted_indices]
self.values = [self.values[i] for i in sorted_indices]
```

1. 按性价比（价值/重量）降序排序
2. 保存原始顺序以便恢复
3. 重新排列重量和价值数组

2.5 算法主循环

```python
best_value = 0
best_selection = []
root = Node(-1, 0, 0, 0, [])
root.bound = calculate_bound(root)
queue = [root]

while queue and time.perf_counter() - start_time <= self.max_time:
    current = queue.pop(0)
    
    if current.bound <= best_value:
        continue
        
    level = current.level + 1
    if level >= self.n:
        continue
    
    # 不选择当前物品
    not_take = Node(level, current.value, current.weight, 0, current.selected)
    not_take.bound = calculate_bound(not_take)
    if not_take.bound > best_value:
        queue.append(not_take)
    
    # 选择当前物品
    new_weight = current.weight + self.weights[level]
    if new_weight <= self.capacity:
        take = Node(level, 
                  current.value + self.values[level],
                  new_weight, 0, current.selected)
        take.selected.append(sorted_indices[level])
        take.bound = calculate_bound(take)
        if take.value > best_value:
            best_value = take.value
            best_selection = take.selected.copy()
        if take.bound > best_value:
            queue.append(take)
    
    queue.sort(key=lambda x: x.bound, reverse=True)
    if len(queue) > 1000:
        queue = queue[:1000]
```

算法执行流程：
1. 初始化最优解和根节点
2. 主循环处理队列中的节点：
   • 取出当前最优节点（上界最大）

   • 剪枝：如果上界≤当前最优值，跳过

   • 生成"不选"和"选"两个子节点

   • 更新最优解

   • 维护优先队列（按上界排序）

   • 限制队列大小（防止内存爆炸）


2.6 恢复数据和返回结果

```python
# 恢复原始数据
self.weights = original_weights
self.values = original_values

end_time = time.perf_counter()
return best_value, end_time - start_time, sorted(best_selection)
```

1. 恢复原始物品顺序
2. 返回最优解和计算时间

3. 算法分析

3.1 时间复杂度

• 最坏情况：O(2^n)（与回溯法相同）

• 平均情况：远好于回溯法，取决于剪枝效果

• 排序：O(n log n)


3.2 空间复杂度

• 队列存储：O(b^d)，b是分支因子，d是最大深度

• 剪枝控制：通过队列大小限制控制内存使用


3.3 优点

1. 比回溯法更高效（通过上界剪枝）
2. 能找到精确最优解
3. 通过优先队列优化搜索顺序

3.4 局限性

1. 最坏情况下时间复杂度仍较高
2. 实现相对复杂
3. 需要合理设置队列大小限制

4. 执行过程示例

假设输入（已排序）：
• 容量=10

• 物品：

  • 物品0：重量=2，价值=6（性价比3.0）

  • 物品1：重量=4，价值=10（性价比2.5）

  • 物品2：重量=5，价值=9（性价比1.8）


搜索过程：
1. 根节点：bound=3*10=30
2. 选择物品0：
   • 价值=6，重量=2

   • bound=6 + 10 + (10-2-4)*1.8=6+10+7.2=23.2

3. 不选物品0：
   • bound=10 + (10-4)*1.8=10+10.8=20.8

4. 优先探索bound=23.2的节点
5. 继续分支和剪枝，最终找到最优解

5. 优化方向

1. 更好的上界计算：使用更精确的上界估计
2. 并行处理：同时处理多个有希望的节点
3. 启发式策略：结合其他启发式方法优化搜索
4. 动态队列大小：根据问题规模调整队列限制

这段代码展示了分支限界法在0-1背包问题中的高效实现，通过智能剪枝和优先搜索策略，在保证找到最优解的同时显著提高了搜索效率。