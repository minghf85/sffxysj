import random
from Maxbackpackvalue import MaxBackpackValue
from typing import List, Tuple

def generate_test_data(n: int) -> tuple:
    """生成测试数据"""
    random.seed(25)  # 固定随机种子以便复现结果
    weights = [random.randint(1, 100) for _ in range(n)]
    values = [random.randint(1, 100) for _ in range(n)]
    capacity = sum(weights) // 3  # 设置背包容量为总重量的1/3
    return weights, values, capacity

def print_solution(name: str, result: Tuple[int, float, List[int]], weights: List[int], values: List[int], capacity: int):
    """打印解决方案"""
    value, time_taken, selected = result
    print(f"\n{name}：")
    print(f"最优解值: {value}")
    print(f"求解时间: {time_taken:.9f}秒")
    if value != -1:  # 如果没有超时
        total_weight = sum(weights[i] for i in selected)
        print(f"选择的物品: {selected}")
        print(f"总重量: {total_weight}/{capacity}")
        print(f"解的验证: {'有效' if total_weight <= capacity else '无效'}")

def test_scale(n: int) -> None:
    """测试特定规模的数据"""
    weights, values, capacity = generate_test_data(n)
    solver = MaxBackpackValue(weights, values, capacity)
    
    print(f"\n{'='*50}")
    print(f"测试数据规模: n = {n}")
    print(f"背包容量: {capacity}")
    print(f"物品重量范围: [{min(weights)}, {max(weights)}]")
    print(f"物品价值范围: [{min(values)}, {max(values)}]")
    print(f"{'='*50}")

    results = []
    
    # 对于大规模数据，跳过某些算法
    if n <= 20:
        result = solver.divide_and_conquer()
        print_solution("1. 分治法", result, weights, values, capacity)
        results.append(("分治法", result))

    result = solver.dynamic_programming()
    print_solution("2. 动态规划", result, weights, values, capacity)
    results.append(("动态规划", result))

    result = solver.greedy()
    print_solution("3. 贪心算法", result, weights, values, capacity)
    results.append(("贪心算法", result))

    if n <= 20:
        result = solver.backtracking()
        print_solution("4. 回溯法", result, weights, values, capacity)
        results.append(("回溯法", result))

    if n <= 50:
        result = solver.branch_and_bound()
        print_solution("5. 分支限界法", result, weights, values, capacity)
        results.append(("分支限界法", result))

    # 比较结果
    print("\n算法比较：")
    print(f"{'算法名称':<12} {'最优值':<8} {'时间(秒)':<15} {'是否最优解':<10}")
    print("-" * 45)
    
    # 找出最优值
    best_value = max(r[1][0] for r in results if r[1][0] != -1)
    
    for name, (value, time_taken, _) in results:
        is_optimal = "是" if value == best_value else "否"
        if value == -1:
            is_optimal = "超时"
        print(f"{name:<12} {value:<8} {time_taken:<15.9f} {is_optimal:<10}")

def main():
    # 测试不同规模的数据
    print("\nn=10:")
    test_scale(10)
    
    print("\nn=50:")
    test_scale(50)
    
    print("\nn=200:")
    test_scale(200)

if __name__ == "__main__":
    main()
