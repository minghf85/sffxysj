#include <iostream>
#include <algorithm>
#include <vector>

// 二叉搜索树节点
struct BSTNode {
    double value;
    BSTNode* left;
    BSTNode* right;
    BSTNode(double val) : value(val), left(nullptr), right(nullptr) {}
};

// AVL树节点
struct AVLNode {
    double value;
    AVLNode* left;
    AVLNode* right;
    int height;
    AVLNode(double val) : value(val), left(nullptr), right(nullptr), height(1) {}
};

// 红黑树节点
struct RBNode {
    double value;
    RBNode* left;
    RBNode* right;
    RBNode* parent;
    bool isRed;
    RBNode(double val) : value(val), left(nullptr), right(nullptr), parent(nullptr), isRed(true) {}
};

// B树节点
struct BTreeNode {
    std::vector<double> keys;
    std::vector<BTreeNode*> children;
    bool isLeaf;
    BTreeNode(bool leaf = true) : isLeaf(leaf) {}
};

// 二叉搜索树类
class BSTree {
private:
    BSTNode* root;
    
    BSTNode* insert(BSTNode* node, double value);
    BSTNode* remove(BSTNode* node, double value);
    BSTNode* search(BSTNode* node, double value);
    void cleanup(BSTNode* node);
    
public:
    BSTree() : root(nullptr) {}
    ~BSTree() { cleanup(root); }
    
    void insert(double value) { root = insert(root, value); }
    void remove(double value) { root = remove(root, value); }
    bool search(double value) { return search(root, value) != nullptr; }
    BSTNode* getRoot() { return root; }
};

// AVL树类
class AVLTree {
private:
    AVLNode* root;
    
    int height(AVLNode* node);
    int getBalance(AVLNode* node);
    AVLNode* rightRotate(AVLNode* y);
    AVLNode* leftRotate(AVLNode* x);
    AVLNode* insert(AVLNode* node, double value);
    AVLNode* remove(AVLNode* node, double value);
    AVLNode* search(AVLNode* node, double value);
    void cleanup(AVLNode* node);
    
public:
    AVLTree() : root(nullptr) {}
    ~AVLTree() { cleanup(root); }
    
    void insert(double value) { root = insert(root, value); }
    void remove(double value) { root = remove(root, value); }
    bool search(double value) { return search(root, value) != nullptr; }
    AVLNode* getRoot() { return root; }
};

// 红黑树类
class RBTree {
private:
    RBNode* root;
    RBNode* NIL;
    
    void leftRotate(RBNode* x);
    void rightRotate(RBNode* y);
    void insertFixup(RBNode* z);
    void deleteFixup(RBNode* x);
    void transplant(RBNode* u, RBNode* v);
    RBNode* minimum(RBNode* node);//最小值
    RBNode* search(RBNode* node, double value);
    void cleanup(RBNode* node);
    
public:
    RBTree();
    ~RBTree();
    
    void insert(double value);
    void remove(double value);
    bool search(double value);
    RBNode* getRoot() { return root; }
};

// B树类
class BTree {
private:
    BTreeNode* root;
    int M;  // 阶数，一个节点最多可以有M个子节点，M-1个键
    
    void splitChild(BTreeNode* x, int i);
    void insertNonFull(BTreeNode* x, double k);
    bool searchInternal(BTreeNode* x, double k);
    void cleanup(BTreeNode* node);
    void removeInternal(BTreeNode* node, double value);
    double getPredecessor(BTreeNode* node);
    double getSuccessor(BTreeNode* node);
    void merge(BTreeNode* parent, int idx);
    void borrowFromLeft(BTreeNode* parent, int idx);
    void borrowFromRight(BTreeNode* parent, int idx);
    
public:
    BTree(int order) : root(nullptr), M(order) {}
    ~BTree() { cleanup(root); }
    
    void insert(double value);
    void remove(double value);
    bool search(double value);
    BTreeNode* getRoot() { return root; }
};

// BST实现
BSTNode* BSTree::insert(BSTNode* node, double value) {
    if (node == nullptr) {
        return new BSTNode(value);
    }
    
    if (value < node->value) {
        node->left = insert(node->left, value);
    } else if (value > node->value) {
        node->right = insert(node->right, value);
    }
    
    return node;
}

BSTNode* BSTree::search(BSTNode* node, double value) {
    if (node == nullptr || node->value == value) {
        return node;
    }
    
    if (value < node->value) {
        return search(node->left, value);
    }
    return search(node->right, value);
}

void BSTree::cleanup(BSTNode* node) {
    if (node != nullptr) {
        cleanup(node->left);
        cleanup(node->right);
        delete node;
    }
}

BSTNode* BSTree::remove(BSTNode* node, double value) {
    if (node == nullptr) return nullptr;
    
    if (value < node->value) {
        node->left = remove(node->left, value);
    } else if (value > node->value) {
        node->right = remove(node->right, value);
    } else {
        if (node->left == nullptr) {//左子树为空，右子树代替(包括了左右子树都为空的情况)
            BSTNode* temp = node->right;
            delete node;
            return temp;
        } else if (node->right == nullptr) {//右子树为空，左子树代替(包括了左右子树都为空的情况)
            BSTNode* temp = node->left;
            delete node;
            return temp;
        }
        //直接前驱或者直接后继
        //找到右子树中的最小值
        BSTNode* temp = node->right;
        while (temp->left != nullptr) {
            temp = temp->left;
        }
        node->value = temp->value;
        node->right = remove(node->right, temp->value);
        //找到左子树中的最大值
        // BSTNode* temp = node->left;
        // while (temp->right != nullptr) {
        //     temp = temp->right;
        // }
        // node->value = temp->value;
        // node->left = remove(node->left, temp->value);
    }
    return node;
}

// AVL树实现
int AVLTree::height(AVLNode* node) {
    if (node == nullptr) return 0;
    return node->height;
}

int AVLTree::getBalance(AVLNode* node) {
    if (node == nullptr) return 0;
    return height(node->left) - height(node->right);
}

AVLNode* AVLTree::rightRotate(AVLNode* y) {
    AVLNode* x = y->left;//失衡节点的左子树
    AVLNode* T2 = x->right;//如果冲突需要调整的节点
    
    x->right = y;
    y->left = T2;//冲突的右子树变为失衡节点的左子树
    //只会影响x,y的高度,更新高度
    y->height = std::max(height(y->left), height(y->right)) + 1;
    x->height = std::max(height(x->left), height(x->right)) + 1;
    
    return x;
}

AVLNode* AVLTree::leftRotate(AVLNode* x) {
    AVLNode* y = x->right;//失衡节点的右子树
    AVLNode* T2 = y->left;//如果冲突需要调整的节点
    
    y->left = x;
    x->right = T2;//冲突的左子树变为失衡节点的右子树
    //只会影响x,y的高度,更新高度
    x->height = std::max(height(x->left), height(x->right)) + 1;
    y->height = std::max(height(y->left), height(y->right)) + 1;
    
    return y;
}

AVLNode* AVLTree::insert(AVLNode* node, double value) {
    if (node == nullptr) {
        return new AVLNode(value);
    }
    
    if (value < node->value) {
        node->left = insert(node->left, value);
    } else if (value > node->value) {
        node->right = insert(node->right, value);
    } else {
        return node;
    }
    
    node->height = 1 + std::max(height(node->left), height(node->right));
    
    int balance = getBalance(node);
    
    // 左左情况
    if (balance > 1 && value < node->left->value) {
        return rightRotate(node);
    }
    
    // 右右情况
    if (balance < -1 && value > node->right->value) {
        return leftRotate(node);
    }
    
    // 左右情况：插入节点位置在失衡节点的左子树的右子树上
    if (balance > 1 && value > node->left->value) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }
    
    // 右左情况：插入节点位置在失衡节点的右子树的左子树上
    if (balance < -1 && value < node->right->value) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }
    
    return node;
}

AVLNode* AVLTree::search(AVLNode* node, double value) {
    if (node == nullptr || node->value == value) {
        return node;
    }
    
    if (value < node->value) {
        return search(node->left, value);
    }
    return search(node->right, value);
}

void AVLTree::cleanup(AVLNode* node) {
    if (node != nullptr) {
        cleanup(node->left);
        cleanup(node->right);
        delete node;
    }
}

// AVL树删除实现
AVLNode* AVLTree::remove(AVLNode* node, double value) {
    if (node == nullptr) return nullptr;
    
    if (value < node->value) {
        node->left = remove(node->left, value);
    } else if (value > node->value) {
        node->right = remove(node->right, value);
    } else {
        if (node->left == nullptr || node->right == nullptr) {
            AVLNode* temp = node->left ? node->left : node->right;//temp指向非空的子树或者nullptr(左右都空)
            
            if (temp == nullptr) {
                temp = node;
                node = nullptr;
            } else {
                *node = *temp;//用非空的子树替换要删除的节点
            }
            delete temp;
        } else {//左右子树都非空,选择右子树的最小值替换
            AVLNode* temp = node->right;
            while (temp->left != nullptr) {
                temp = temp->left;
            }
            node->value = temp->value;
            node->right = remove(node->right, temp->value);
            //左右子树都非空,选择左子树的最大值替换
            // AVLNode* temp = node->left;
            // while (temp->right != nullptr) {
            //     temp = temp->right;
            // }
            // node->value = temp->value;
            // node->left = remove(node->left, temp->value);
        }
    }
    
    if (node == nullptr) return nullptr;
    
    node->height = 1 + std::max(height(node->left), height(node->right));//更新高度,检查是否存在失衡
    
    int balance = getBalance(node);
    
    // 左左情况
    if (balance > 1 && getBalance(node->left) >= 0) {
        return rightRotate(node);
    }
    
    // 左右情况
    if (balance > 1 && getBalance(node->left) < 0) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }
    
    // 右右情况
    if (balance < -1 && getBalance(node->right) <= 0) {
        return leftRotate(node);
    }
    
    // 右左情况
    if (balance < -1 && getBalance(node->right) > 0) {
        node->right = rightRotate(node->right);
        return leftRotate(node);
    }
    
    return node;
}

// 红黑树实现
RBTree::RBTree() {
    NIL = new RBNode(0);
    NIL->isRed = false;
    root = NIL;
}

RBTree::~RBTree() {
    cleanup(root);
    delete NIL;
}

void RBTree::leftRotate(RBNode* x) {
    RBNode* y = x->right;
    x->right = y->left;
    
    if (y->left != NIL) {
        y->left->parent = x;
    }
    
    y->parent = x->parent;
    
    if (x->parent == nullptr) {
        root = y;
    } else if (x == x->parent->left) {
        x->parent->left = y;
    } else {
        x->parent->right = y;
    }
    
    y->left = x;
    x->parent = y;
}

void RBTree::rightRotate(RBNode* y) {
    RBNode* x = y->left;
    y->left = x->right;
    
    if (x->right != NIL) {
        x->right->parent = y;
    }
    
    x->parent = y->parent;
    
    if (y->parent == nullptr) {
        root = x;
    } else if (y == y->parent->right) {
        y->parent->right = x;
    } else {
        y->parent->left = x;
    }
    
    x->right = y;
    y->parent = x;
}

void RBTree::insert(double value) {
    //在叶子节点上插入新的红色节点，最后再调整
    RBNode* z = new RBNode(value);
    RBNode* y = nullptr;
    RBNode* x = root;
    
    while (x != NIL) {
        y = x;
        if (z->value < x->value) {
            x = x->left;
        } else {
            x = x->right;
        }
    }
    
    z->parent = y;
    
    if (y == nullptr) {
        root = z;
    } else if (z->value < y->value) {
        y->left = z;
    } else {
        y->right = z;
    }
    
    z->left = NIL;
    z->right = NIL;
    z->isRed = true;
    
    insertFixup(z);
}

void RBTree::insertFixup(RBNode* z) {
    while (z->parent != nullptr && z->parent->isRed) {//下面如果z = z->parent->parent;到了根节点，会跳出调整为黑色；违反不红红则会继续
        if (z->parent == z->parent->parent->left) {//如果父节点是爷爷的左子树
            RBNode* y = z->parent->parent->right;//叔叔节点
            if (y->isRed) {//叔叔是红色节点，则父、叔、爷变色
                z->parent->isRed = false;
                y->isRed = false;
                z->parent->parent->isRed = true;
                z = z->parent->parent;//爷爷变插入节点，只有这个地方会向上调整，一直循环！！！！！！
            } else {//叔叔是黑色节点或者空节点，即黑色节点
                if (z == z->parent->right) {//LR情况，先左旋转
                    z = z->parent;
                    leftRotate(z);
                }
                z->parent->isRed = false;
                z->parent->parent->isRed = true;
                rightRotate(z->parent->parent);//爷爷作为旋转点右旋
            }
        } else {//如果父节点是爷爷的右子树，同理反过来判断和处理
            RBNode* y = z->parent->parent->left;
            if (y->isRed) {
                z->parent->isRed = false;
                y->isRed = false;
                z->parent->parent->isRed = true;
                z = z->parent->parent;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    rightRotate(z);
                }
                z->parent->isRed = false;
                z->parent->parent->isRed = true;
                leftRotate(z->parent->parent);
            }
        }
    }
    root->isRed = false;//如果根节点是红色，变为黑色
}

bool RBTree::search(double value) {
    return search(root, value) != NIL;
}

RBNode* RBTree::search(RBNode* node, double value) {
    if (node == NIL || value == node->value) {
        return node;
    }
    
    if (value < node->value) {
        return search(node->left, value);
    }
    return search(node->right, value);
}

void RBTree::cleanup(RBNode* node) {
    if (node != NIL) {
        cleanup(node->left);
        cleanup(node->right);
        delete node;
    }
}

void RBTree::remove(double value) {
    // 查找要删除的节点
    RBNode* z = search(root, value);
    if (z == NIL) return; // 节点不存在，直接返回
    
    RBNode* y = z; // y 指向将要被删除或移动的节点
    RBNode* x;     // x 记录 y 原来的位置，用于后续调整
    bool yOriginalColor = y->isRed; // 保存 y 的原始颜色
    
    // 处理三种情况：z 无左子、无右子、有双子节点
    if (z->left == NIL) {          // 情况1：z 无左子节点
        x = z->right;
        transplant(z, z->right);   // 用右子替换 z
    } else if (z->right == NIL) {  // 情况2：z 无右子节点
        x = z->left;
        transplant(z, z->left);    // 用左子替换 z
    } else {                       // 情况3：z 有双子节点
        y = minimum(z->right);      // 找到 z 的后继节点 y（右子树的最小节点）
        yOriginalColor = y->isRed;  // 更新为 y 的原始颜色
        x = y->right;               // x 指向 y 的右子（可能是 NIL）
        
        // 处理 y 的父节点关系
        if (y->parent == z) {       // 情况3a：y 是 z 的直接右子
            x->parent = y;          // 确保 x 的父节点正确（即使 x 是 NIL）
        } else {                    // 情况3b：y 不是 z 的直接右子
            transplant(y, y->right);// 用 y 的右子替换 y
            y->right = z->right;    // 将 z 的右子树接到 y
            y->right->parent = y;
        }
        
        transplant(z, y);           // 用 y 替换 z
        y->left = z->left;          // 将 z 的左子树接到 y
        y->left->parent = y;
        y->isRed = z->isRed;        // 继承 z 的颜色
    }
    
    // 如果被删除的节点 y 是黑色，需要调整红黑树性质
    if (!yOriginalColor) {
        deleteFixup(x); // 从 x 开始修复
    }
    
    delete z; // 释放节点内存
}

// 用节点 v 替换节点 u 的位置（仅处理父子关系，不处理子节点）
void RBTree::transplant(RBNode* u, RBNode* v) {
    if (u->parent == nullptr) {     // u 是根节点
        root = v;
    } else if (u == u->parent->left) { // u 是左子节点
        u->parent->left = v;
    } else {                        // u 是右子节点
        u->parent->right = v;
    }
    v->parent = u->parent;          // 更新 v 的父节点
}

// 找到以 node 为根的子树的最小节点（最左子节点）
RBNode* RBTree::minimum(RBNode* node) {
    while (node->left != NIL) {
        node = node->left;
    }
    return node;
}

// 修复删除后的红黑树性质
void RBTree::deleteFixup(RBNode* x) {
    // 循环处理直到 x 是根或变为红色
    while (x != root && !x->isRed) {
        if (x == x->parent->left) { // x 是左子节点
            RBNode* w = x->parent->right; // 兄弟节点
            if (w->isRed) {         // 情况1：兄弟是红色
                w->isRed = false;   // 兄弟变黑
                x->parent->isRed = true; // 父变红
                leftRotate(x->parent);  // 左旋父节点，转化为兄弟为黑的情况
                w = x->parent->right; // 更新兄弟节点
            }
            // 兄弟节点为黑色时的处理
            if (!w->left->isRed && !w->right->isRed) { // 情况2：兄弟的两个子都是黑色
                w->isRed = true;    // 兄弟变红
                x = x->parent;      // 将 x 上移，继续循环处理父节点
            } else {
                if (!w->right->isRed) { // 情况3：兄弟的右子是黑色（左子为红）
                    w->left->isRed = false; // 左子变黑
                    w->isRed = true;    // 兄弟变红
                    rightRotate(w);     // 右旋兄弟节点，转化为情况4
                    w = x->parent->right; // 更新兄弟节点
                }
                // 情况4：兄弟的右子是红色
                w->isRed = x->parent->isRed; // 兄弟颜色继承父节点颜色
                x->parent->isRed = false;    // 父节点变黑
                w->right->isRed = false;     // 兄弟右子变黑
                leftRotate(x->parent);       // 左旋父节点
                x = root;                    // 结束循环（x 设为根）
            }
        } else { // 对称处理：x 是右子节点
            RBNode* w = x->parent->left;
            if (w->isRed) {
                w->isRed = false;
                x->parent->isRed = true;
                rightRotate(x->parent);
                w = x->parent->left;
            }
            if (!w->right->isRed && !w->left->isRed) {
                w->isRed = true;
                x = x->parent;
            } else {
                if (!w->left->isRed) {
                    w->right->isRed = false;
                    w->isRed = true;
                    leftRotate(w);
                    w = x->parent->left;
                }
                w->isRed = x->parent->isRed;
                x->parent->isRed = false;
                w->left->isRed = false;
                rightRotate(x->parent);
                x = root;
            }
        }
    }
    x->isRed = false; // 确保根节点为黑色
}

// B树实现
void BTree::insert(double value) {
    if (root == nullptr) {
        root = new BTreeNode(true);
        root->keys.push_back(value);
        return;
    }
    
    // 如果根节点已满，需要分裂
    if (root->keys.size() == M - 1) {//如果插入就变成M溢出了
        BTreeNode* newRoot = new BTreeNode(false);
        newRoot->children.push_back(root);
        // 分裂原根节点(现在位于newRoot->children[0])
        splitChild(newRoot, 0);//根节点第一次是0
        root = newRoot;
    }
    insertNonFull(root, value);
}

void BTree::splitChild(BTreeNode* x, int i) {
    BTreeNode* y = x->children[i];  // 获取要分裂的子节点y
    BTreeNode* z = new BTreeNode(y->isLeaf); // 创建新节点z
    
    int mid = (M - 1) / 2;  // 计算中间位置
    
    // 1. 将y的后半部分键转移到z
    for (int j = mid + 1; j < y->keys.size(); j++) {
        z->keys.push_back(y->keys[j]); // 复制y的后半部分键
    }
    
    // 2. 如果y不是叶子节点，还需转移子节点指针
    if (!y->isLeaf) {
        for (int j = mid + 1; j < y->children.size(); j++) {
            z->children.push_back(y->children[j]); // 复制y的后半部分子节点
        }
    }
    
    // 3. 将y的中间键(mid索引)提升到父节点x
    x->keys.insert(x->keys.begin() + i, y->keys[mid]);
    
    // 4. 将新节点z插入父节点x的子节点列表(i+1位置)
    //前半部分的位置不用改，仍然是父节点x的子节点列表的i位置处
    x->children.insert(x->children.begin() + i + 1, z);
    
    // 5. 调整y的大小，保留前mid个键
    y->keys.resize(mid);
    // 如果是内部节点，保留前mid+1个子节点
    if (!y->isLeaf) {
        y->children.resize(mid + 1);
    }
}

void BTree::insertNonFull(BTreeNode* x, double k) {
    int i = x->keys.size() - 1;
    
    if (x->isLeaf) {
        // 在叶子节点中插入键
        while (i >= 0 && k < x->keys[i]) {
            i--;
        }
        x->keys.insert(x->keys.begin() + i + 1, k);
    } else {
        // 找到合适的子节点对应的i，即位置
        while (i >= 0 && k < x->keys[i]) {
            i--;
        }
        i++;
        
        // 检查子节点是否需要分裂
        if (x->children[i]->keys.size() == M - 1) {//如果插入就变成M溢出了
            splitChild(x, i);
            if (k > x->keys[i]) {
                i++;//分裂之后会提上来一个新的子节点，插入值位置要变成i+1
            }
        }
        insertNonFull(x->children[i], k);//找到合适的范围子树位置
    }
}

bool BTree::search(double value) {
    return searchInternal(root, value);
}

bool BTree::searchInternal(BTreeNode* x, double k) {
    if (x == nullptr) return false;
    
    int i = 0;
    while (i < x->keys.size() && k > x->keys[i]) {
        i++;
    }
    
    if (i < x->keys.size() && k == x->keys[i]) {
        return true;
    }
    
    if (x->isLeaf) {
        return false;
    }
    
    return searchInternal(x->children[i], k);//在当前节点范围内，但没有相等的
}

void BTree::cleanup(BTreeNode* node) {
    if (node != nullptr) {
        if (!node->isLeaf) {
            for (BTreeNode* child : node->children) {
                cleanup(child);
            }
        }
        delete node;
    }
}

// B树删除实现
void BTree::remove(double value) {
    if (root == nullptr) return;
    
    removeInternal(root, value);
    
    // 如果根节点变空，需要调整树的高度
    if (root->keys.empty()) {
        if (root->isLeaf) {
            delete root;
            root = nullptr;
        } else {
            BTreeNode* oldRoot = root;
            root = root->children[0];
            delete oldRoot;
        }
    }
}

void BTree::removeInternal(BTreeNode* node, double value) {
    if (node == nullptr) return;
    
    int idx = 0;
    while (idx < node->keys.size() && value > node->keys[idx]) {
        idx++;
    }
    
    if (idx < node->keys.size() && value == node->keys[idx]) {
        // 找到要删除的键
        if (node->isLeaf) {
            // 如果是叶子节点，直接删除
            node->keys.erase(node->keys.begin() + idx);
        } else {
            // 如果不是叶子节点
            BTreeNode* leftChild = node->children[idx];
            BTreeNode* rightChild = node->children[idx + 1];
            
            if (leftChild->keys.size() >= (M - 1) / 2) {
                // 如果左子节点有足够的键，用前驱替换
                double predecessor = getPredecessor(leftChild);
                node->keys[idx] = predecessor;
                removeInternal(leftChild, predecessor);
            } else if (rightChild->keys.size() >= (M - 1) / 2) {
                // 如果右子节点有足够的键，用后继替换
                double successor = getSuccessor(rightChild);
                node->keys[idx] = successor;
                removeInternal(rightChild, successor);
            } else {
                // 如果左右子节点都没有足够的键，合并它们
                merge(node, idx);
                removeInternal(leftChild, value);
            }
        }
    } else {
        // 键不在当前节点
        if (node->isLeaf) return;  // 键不存在
        
        BTreeNode* child = node->children[idx];
        bool isLastChild = (idx == node->keys.size());
        
        // 确保子节点至少有最小数量的键
        if (child->keys.size() < (M - 1) / 2) {
            bool merged = false;
            
            // 尝试从左兄弟借
            if (idx > 0 && node->children[idx - 1]->keys.size() >= (M - 1) / 2) {
                borrowFromLeft(node, idx);
            }
            // 尝试从右兄弟借
            else if (idx < node->keys.size() && node->children[idx + 1]->keys.size() >= (M - 1) / 2) {
                borrowFromRight(node, idx);
            }
            // 需要合并
            else {
                if (idx > 0) {
                    // 与左兄弟合并
                    merge(node, idx - 1);
                    child = node->children[idx - 1];
                    idx--;
                } else {
                    // 与右兄弟合并
                    merge(node, idx);
                }
                merged = true;
            }
            
            // 如果发生了合并且是最后一个子节点，需要调整索引
            if (merged && isLastChild && idx > 0) {
                idx--;
            }
        }
        
        removeInternal(node->children[idx], value);
    }
}

double BTree::getPredecessor(BTreeNode* node) {
    while (!node->isLeaf) {
        node = node->children.back();
    }
    return node->keys.back();
}

double BTree::getSuccessor(BTreeNode* node) {
    while (!node->isLeaf) {
        node = node->children.front();
    }
    return node->keys.front();
}

void BTree::merge(BTreeNode* parent, int idx) {
    BTreeNode* leftChild = parent->children[idx];
    BTreeNode* rightChild = parent->children[idx + 1];
    
    // 将父节点的键移到左子节点
    leftChild->keys.push_back(parent->keys[idx]);
    
    // 将右子节点的键移到左子节点
    leftChild->keys.insert(leftChild->keys.end(), rightChild->keys.begin(), rightChild->keys.end());
    
    // 如果不是叶子节点，移动子节点
    if (!leftChild->isLeaf) {
        leftChild->children.insert(leftChild->children.end(), rightChild->children.begin(), rightChild->children.end());
    }
    
    // 从父节点中删除键和右子节点
    parent->keys.erase(parent->keys.begin() + idx);
    parent->children.erase(parent->children.begin() + idx + 1);
    
    delete rightChild;
}

void BTree::borrowFromLeft(BTreeNode* parent, int idx) {
    BTreeNode* child = parent->children[idx];
    BTreeNode* leftSibling = parent->children[idx - 1];
    
    // 将父节点的键移到子节点
    child->keys.insert(child->keys.begin(), parent->keys[idx - 1]);
    
    // 将左兄弟的最后一个键移到父节点
    parent->keys[idx - 1] = leftSibling->keys.back();
    leftSibling->keys.pop_back();
    
    // 如果不是叶子节点，移动子节点
    if (!child->isLeaf) {
        child->children.insert(child->children.begin(), leftSibling->children.back());
        leftSibling->children.pop_back();
    }
}

void BTree::borrowFromRight(BTreeNode* parent, int idx) {
    BTreeNode* child = parent->children[idx];
    BTreeNode* rightSibling = parent->children[idx + 1];
    
    // 将父节点的键移到子节点
    child->keys.push_back(parent->keys[idx]);
    
    // 将右兄弟的第一个键移到父节点
    parent->keys[idx] = rightSibling->keys.front();
    rightSibling->keys.erase(rightSibling->keys.begin());
    
    // 如果不是叶子节点，移动子节点
    if (!child->isLeaf) {
        child->children.push_back(rightSibling->children.front());
        rightSibling->children.erase(rightSibling->children.begin());
    }
}
