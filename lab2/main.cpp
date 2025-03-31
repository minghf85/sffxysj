#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "tree.cpp"

// 生成随机数据
std::vector<double> generateRandomData(int n) {
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1000000);
    
    for (int i = 0; i < n; i++) {
        data.push_back(dis(gen));
    }
    return data;
}

// 测试树的性能
void testTreePerformance(int dataSize) {
    std::vector<double> data = generateRandomData(dataSize);
    std::vector<double> searchData = generateRandomData(dataSize / 10); // 用于搜索测试
    
    BSTree bst;
    AVLTree avl;
    RBTree rb;
    BTree btree(5);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    
    // 测试插入性能
    std::cout << "\n" << std::endl;
    std::cout << "\n数据规模: " << dataSize << std::endl;
    std::cout << "\n=== 插入性能测试 ===" << std::endl;
    
    // BST插入测试
    start = std::chrono::high_resolution_clock::now();
    for (double value : data) {
        bst.insert(value);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "BST插入时间: " << std::fixed << std::setprecision(6) 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << std::endl;
    
    // AVL插入测试
    start = std::chrono::high_resolution_clock::now();
    for (double value : data) {
        avl.insert(value);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "AVL插入时间: " << std::fixed << std::setprecision(6) 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << std::endl;
    
    // 红黑树插入测试
    start = std::chrono::high_resolution_clock::now();
    for (double value : data) {
        rb.insert(value);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "红黑树插入时间: " << std::fixed << std::setprecision(6) 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << std::endl;
    
    // B树插入测试
    start = std::chrono::high_resolution_clock::now();
    for (double value : data) {
        btree.insert(value);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "B树插入时间: " << std::fixed << std::setprecision(6) 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << std::endl;
    
    // 测试搜索性能
    std::cout << "\n=== 搜索性能测试 ===" << std::endl;
    
    // BST搜索测试
    start = std::chrono::high_resolution_clock::now();
    for (double value : searchData) {
        bst.search(value);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "BST搜索时间: " << std::fixed << std::setprecision(6) 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << std::endl;
    
    // AVL搜索测试
    start = std::chrono::high_resolution_clock::now();
    for (double value : searchData) {
        avl.search(value);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "AVL搜索时间: " << std::fixed << std::setprecision(6) 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << std::endl;
    
    // 红黑树搜索测试
    start = std::chrono::high_resolution_clock::now();
    for (double value : searchData) {
        rb.search(value);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "红黑树搜索时间: " << std::fixed << std::setprecision(6) 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << std::endl;
    
    // B树搜索测试
    start = std::chrono::high_resolution_clock::now();
    for (double value : searchData) {
        btree.search(value);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "B树搜索时间: " << std::fixed << std::setprecision(6) 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << std::endl;
}

int main() {
    std::vector<int> sizes = {1000, 10000, 100000};
    
    for (int size : sizes) {
        testTreePerformance(size);
    }
    
    return 0;
}
