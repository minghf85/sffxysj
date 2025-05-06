import time
from typing import List, Tuple

class MaxBackpackValue:
    def __init__(self, weights: List[int], values: List[int], capacity: int):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n = len(weights)
        self.max_time = 30  # 设置最大运行时间为60秒
        self.selected_items = []  # 记录选择的物品

    def divide_and_conquer(self) -> Tuple[int, float, List[int]]:
        """分治法求解0-1背包问题
        Returns:
            tuple: (最优值, 计算时间, 选择的物品列表)
        """
        #初始化
        start_time = time.perf_counter()
        self.selected_items = [0] * self.n
        self.best_value = 0
        self.best_selection = [0] * self.n

        def solve(i: int, c: int, current_selection: List[int]) -> int:
            if time.perf_counter() - start_time > self.max_time:
                return -1
            if i == 0 or c == 0:#递归终止条件
                current_value = sum(v * s for v, s in zip(self.values, current_selection))
                if current_value > self.best_value:
                    self.best_value = current_value
                    self.best_selection = current_selection.copy()
                return 0

            # 不选择当前物品
            value1 = solve(i - 1, c, current_selection)

            # 选择当前物品
            if self.weights[i - 1] <= c:
                current_selection[i - 1] = 1
                value2 = self.values[i - 1] + solve(i - 1, c - self.weights[i - 1], current_selection)
                current_selection[i - 1] = 0
                return max(value1, value2)
            return value1

        current_selection = [0] * self.n
        result = solve(self.n, self.capacity, current_selection)
        end_time = time.perf_counter()

        # 将选择的物品编号转换为索引列表
        selected_items = [i for i, selected in enumerate(self.best_selection) if selected == 1]
        return result, end_time - start_time, selected_items

    def dynamic_programming(self) -> Tuple[int, float, List[int]]:
        """动态规划求解0-1背包问题"""
        start_time = time.perf_counter()
        dp = [[0] * (self.capacity + 1) for _ in range(self.n + 1)]
        # 用于记录选择
        selected = [[False] * (self.capacity + 1) for _ in range(self.n + 1)]
        
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
        
        # 回溯找出选择的物品
        selected_items = []
        w = self.capacity
        for i in range(self.n, 0, -1):
            if selected[i][w]:
                selected_items.append(i-1)
                w -= self.weights[i-1]
        
        return dp[self.n][self.capacity], time.perf_counter() - start_time, selected_items

    def greedy(self) -> Tuple[int, float, List[int]]:
        """贪心算法求解0-1背包问题"""
        start_time = time.perf_counter()
        
        # 计算性价比并保存原始索引
        items = [(i, self.values[i]/self.weights[i], self.weights[i], self.values[i]) 
                for i in range(self.n) if self.weights[i] > 0]
        # 按性价比降序排序
        items.sort(key=lambda x: x[1], reverse=True)
        
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
        
        return total_value, time.perf_counter() - start_time, sorted(selected_items)

    def backtracking(self) -> Tuple[int, float, List[int]]:
        """回溯法求解0-1背包问题"""
        start_time = time.perf_counter()
        self.max_value = 0
        self.best_selection = []
        current_selection = []
        
        def backtrack(i: int, current_weight: int, current_value: int, selected: List[int]):
            if time.perf_counter() - start_time > self.max_time:
                return
            if i == self.n:
                if current_value > self.max_value:
                    self.max_value = current_value
                    self.best_selection = selected.copy()
                return
            
            # 选择当前物品
            if current_weight + self.weights[i] <= self.capacity:
                selected.append(i)
                backtrack(i + 1, current_weight + self.weights[i], 
                         current_value + self.values[i], selected)
                selected.pop()
            
            # 不选择当前物品
            backtrack(i + 1, current_weight, current_value, selected)
        
        backtrack(0, 0, 0, current_selection)
        return self.max_value, time.perf_counter() - start_time, self.best_selection

    def branch_and_bound(self) -> Tuple[int, float, List[int]]:
        """分支限界法求解0-1背包问题"""
        start_time = time.perf_counter()
        
        class Node:
            def __init__(self, level, value, weight, bound, selected):
                self.level = level
                self.value = value
                self.weight = weight
                self.bound = bound
                self.selected = selected.copy()  # 记录选择的物品
        
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
                value_bound += (self.capacity - weight) * (self.values[level] / self.weights[level])#分数背包思想
            
            return value_bound

        # 按照单位重量价值排序
        items = [(i, self.values[i]/self.weights[i]) for i in range(self.n)]
        items.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [x[0] for x in items]
        
        # 重新排序weights和values
        original_weights = self.weights.copy()
        original_values = self.values.copy()
        self.weights = [self.weights[i] for i in sorted_indices]
        self.values = [self.values[i] for i in sorted_indices]
        
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
        
        # 恢复原始数据
        self.weights = original_weights
        self.values = original_values
        
        end_time = time.perf_counter()
        return best_value, end_time - start_time, sorted(best_selection)

    def verify_solution(self, selected_items: List[int]) -> bool:
        """验证解的正确性"""
        total_weight = sum(self.weights[i] for i in selected_items)
        total_value = sum(self.values[i] for i in selected_items)
        return total_weight <= self.capacity and total_value > 0
