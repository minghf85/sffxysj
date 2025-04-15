#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "tree.cpp"

using namespace std;
// 生成随机数据
vector<double> generateRandomData(int n) {
    vector<double> data;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1000000);
    
    for (int i = 0; i < n; i++) {
        data.push_back(dis(gen));
    }
    return data;
}

// 打印BST树结构
void printBST(BSTNode* root, std::string prefix = "", bool isLeft = false) {
    if (root == nullptr) {
        cout << prefix;
        cout << (isLeft ? "├──" : "└──" );
        cout << "null" << endl;
        return;
    }
    
    cout << prefix;
    cout << (isLeft ? "├──" : "└──" );
    cout << root->value << endl;
    
    if (root->left == nullptr && root->right == nullptr) return;
    
    // 先打印右子树（大于根的值）
    printBST(root->right, prefix + (isLeft ? "│  " : "   "), true);
    // 再打印左子树（小于根的值）
    printBST(root->left, prefix + (isLeft ? "│  " : "   "), false);
}

// 打印AVL树结构
void printAVL(AVLNode* root, std::string prefix = "", bool isLeft = false) {
    if (root == nullptr) {
        cout << prefix;
        cout << (isLeft ? "├──" : "└──" );
        cout << "null" << endl;
        return;
    }
    
    cout << prefix;
    cout << (isLeft ? "├──" : "└──" );
    cout << root->value << "(" << root->height << ")" << endl;
    
    if (root->left == nullptr && root->right == nullptr) return;
    
    // 先打印右子树（大于根的值）
    printAVL(root->right, prefix + (isLeft ? "│  " : "   "), true);
    // 再打印左子树（小于根的值）
    printAVL(root->left, prefix + (isLeft ? "│  " : "   "), false);
}

// 打印红黑树结构
void printRB(RBNode* root, std::string prefix = "", bool isLeft = false) {
    if (root == nullptr || root->value == 0) {  // NIL节点
        cout << prefix;
        cout << (isLeft ? "├──" : "└──" );
        cout << "NIL" << endl;
        return;
    }
    
    cout << prefix;
    cout << (isLeft ? "├──" : "└──" );
    // 使用ANSI转义序列设置红色文本
    if (root->isRed) {
        cout << "\033[31m" << root->value << "\033[0m" << endl;  // 红色节点
    } else {
        cout << root->value << endl;  // 黑色节点
    }
    
    if (root->left == nullptr && root->right == nullptr) return;
    
    // 先打印右子树（大于根的值）
    printRB(root->right, prefix + (isLeft ? "│  " : "   "), true);
    // 再打印左子树（小于根的值）
    printRB(root->left, prefix + (isLeft ? "│  " : "   "), false);
}

// 打印B树结构
void printBTree(BTreeNode* root, std::string prefix = "", bool isLast = true) {
    if (root == nullptr) {
        cout << prefix;
        cout << (isLast ? "└──" : "├──");
        cout << "null" << endl;
        return;
    }
    
    cout << prefix;
    cout << (isLast ? "└──" : "├──");
    cout << "[";
    for (size_t i = 0; i < root->keys.size(); i++) {
        cout << root->keys[i];
        if (i < root->keys.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    
    if (root->isLeaf) return;
    
    // 从后向前遍历子节点，这样较小的子树会显示在下方
    for (int i = root->children.size() - 1; i >= 0; i--) {
        printBTree(root->children[i], 
                  prefix + (isLast ? "    " : "│   "), 
                  i == 0);  // 最后一个子节点（最小的）使用└──
    }
}

// 测试树的功能，插入，搜索，删除
void testTreeFunction(int dataSize) {
    // vector<double> data = {1, 5, 7, 4, 16, 35, 24, 42, 21, 17, 18};
    vector<double> data = generateRandomData(10);
    BSTree bst;
    AVLTree avl;
    RBTree rb;
    BTree btree(5);  // 创建一个5阶B树，每个节点最多4个键，最少2个键
    
    cout << "\n=== 测试数据 ===" << endl;
    for (double value : data) {
        cout << value << " ";
    }
    cout << "\n" << endl;
    
    // 测试BST
    cout << "=== BST树测试 ===" << endl;
    for (double value : data) {
        bst.insert(value);
    }
    cout << "BST树结构：" << endl;
    printBST(bst.getRoot());
    
    // 测试搜索
    cout << "\nBST搜索测试：" << endl;
    for (double value : data) {
        if (!bst.search(value)) {
            cout << "错误：未找到值 " << value << endl;
        }
    }
    cout << "BST搜索测试完成" << endl;
    
    // 测试删除
    cout << "\nBST删除测试：" << endl;
    for (int i = 0; i < dataSize/2; i++) {
        bst.remove(data[i]);
    }
    cout << "删除后的BST树结构：" << endl;
    printBST(bst.getRoot());
    
    // 测试AVL
    cout << "\n=== AVL树测试 ===" << endl;
    for (double value : data) {
        avl.insert(value);
    }
    cout << "AVL树结构：" << endl;
    printAVL(avl.getRoot());
    
    // 测试搜索
    cout << "\nAVL搜索测试：" << endl;
    for (double value : data) {
        if (!avl.search(value)) {
            cout << "错误：未找到值 " << value << endl;
        }
    }
    cout << "AVL搜索测试完成" << endl;
    
    // 测试删除
    cout << "\nAVL删除测试：" << endl;
    for (int i = 0; i < dataSize/2; i++) {
        avl.remove(data[i]);
    }
    cout << "删除后的AVL树结构：" << endl;
    printAVL(avl.getRoot());
    
    // 测试红黑树
    cout << "\n=== 红黑树测试 ===" << endl;
    for (double value : data) {
        rb.insert(value);
    }
    cout << "红黑树结构：" << endl;
    printRB(rb.getRoot());
    
    // 测试搜索
    cout << "\n红黑树搜索测试：" << endl;
    for (double value : data) {
        if (!rb.search(value)) {
            cout << "错误：未找到值 " << value << endl;
        }
    }
    cout << "红黑树搜索测试完成" << endl;
    
    // 测试删除
    cout << "\nRB删除测试：" << endl;
    for (int i = 0; i < dataSize/2; i++) {
        rb.remove(data[i]);
    }
    cout << "删除后的红黑树结构：" << endl;
    printRB(rb.getRoot());
    
    // 测试B树
    cout << "\n=== B树测试 ===" << endl;
    for (double value : data) {
        btree.insert(value);
    }
    cout << "B树结构：" << endl;
    printBTree(btree.getRoot());
    
    // 测试搜索
    cout << "\nB树搜索测试：" << endl;
    for (double value : data) {
        if (!btree.search(value)) {
            cout << "错误：未找到值 " << value << endl;
        }
    }
    cout << "B树搜索测试完成" << endl;
    
    // 测试删除
    cout << "\nB树删除测试：" << endl;
    for (int i = 0; i < dataSize/2; i++) {
        btree.remove(data[i]);
    }
    cout << "删除后的B树结构：" << endl;
    printBTree(btree.getRoot());
}

// 测试树的性能
void testTreePerformance(int dataSize) {
    vector<double> data = generateRandomData(dataSize);
    vector<double> searchData = generateRandomData(dataSize / 10); // 用于搜索测试
    
    BSTree bst;
    AVLTree avl;
    RBTree rb;
    BTree btree(5);  // 创建一个5阶B树
    
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    
    // 测试插入性能
    cout << "\n" << endl;
    cout << "\n数据规模: " << dataSize << endl;
    cout << "\n=== 插入性能测试 ===" << endl;
    
    // BST插入测试
    start = chrono::high_resolution_clock::now();
    for (double value : data) {
        bst.insert(value);
    }
    end = chrono::high_resolution_clock::now();
    cout << "BST插入时间: " << fixed << setprecision(6) 
              << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << endl;
    
    // AVL插入测试
    start = chrono::high_resolution_clock::now();
    for (double value : data) {
        avl.insert(value);
    }
    end = chrono::high_resolution_clock::now();
    cout << "AVL插入时间: " << fixed << setprecision(6) 
              << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << endl;
    
    // 红黑树插入测试
    start = chrono::high_resolution_clock::now();
    for (double value : data) {
        rb.insert(value);
    }
    end = chrono::high_resolution_clock::now();
    cout << "红黑树插入时间: " << fixed << setprecision(6) 
              << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << endl;
    
    // B树插入测试
    start = chrono::high_resolution_clock::now();
    for (double value : data) {
        btree.insert(value);
    }
    end = chrono::high_resolution_clock::now();
    cout << "B树插入时间: " << fixed << setprecision(6) 
              << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << endl;
    
    // 测试搜索性能
    cout << "\n=== 搜索性能测试 ===" << endl;
    
    // BST搜索测试
    start = chrono::high_resolution_clock::now();
    for (double value : searchData) {
        bst.search(value);
    }
    end = chrono::high_resolution_clock::now();
    cout << "BST搜索时间: " << fixed << setprecision(6) 
              << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << endl;
    
    // AVL搜索测试
    start = chrono::high_resolution_clock::now();
    for (double value : searchData) {
        avl.search(value);
    }
    end = chrono::high_resolution_clock::now();
    cout << "AVL搜索时间: " << fixed << setprecision(6) 
              << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << endl;
    
    // 红黑树搜索测试
    start = chrono::high_resolution_clock::now();
    for (double value : searchData) {
        rb.search(value);
    }
    end = chrono::high_resolution_clock::now();
    cout << "红黑树搜索时间: " << fixed << setprecision(6) 
              << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << endl;
    
    // B树搜索测试
    start = chrono::high_resolution_clock::now();
    for (double value : searchData) {
        btree.search(value);
    }
    end = chrono::high_resolution_clock::now();
    cout << "B树搜索时间: " << fixed << setprecision(6) 
              << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0 
              << "s" << endl;
}

int main() {
    // 使用较小的数据规模进行功能测试
    cout << "=== 开始功能测试 ===" << endl;
    testTreeFunction(10);  // 使用10个数据进行测试
    
    cout << "\n=== 开始性能测试 ===" << endl;
    vector<int> sizes = {10000, 100000, 1000000};
    for (int size : sizes) {
        testTreePerformance(size);
    }
    return 0;
}
