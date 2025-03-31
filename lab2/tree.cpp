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
    RBNode* minimum(RBNode* node);
    RBNode* search(RBNode* node, double value);
    void cleanup(RBNode* node);
    
public:
    RBTree();
    ~RBTree();
    
    void insert(double value);
    void remove(double value);
    bool search(double value);
};

// B树类
class BTree {
private:
    BTreeNode* root;
    int t;  // 最小度数
    
    void splitChild(BTreeNode* x, int i);
    void insertNonFull(BTreeNode* x, double k);
    bool searchInternal(BTreeNode* x, double k);
    void cleanup(BTreeNode* node);
    
public:
    BTree(int degree) : root(nullptr), t(degree) {}
    ~BTree() { cleanup(root); }
    
    void insert(double value);
    void remove(double value);
    bool search(double value);
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
        if (node->left == nullptr) {
            BSTNode* temp = node->right;
            delete node;
            return temp;
        } else if (node->right == nullptr) {
            BSTNode* temp = node->left;
            delete node;
            return temp;
        }
        
        BSTNode* temp = node->right;
        while (temp->left != nullptr) {
            temp = temp->left;
        }
        node->value = temp->value;
        node->right = remove(node->right, temp->value);
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
    AVLNode* x = y->left;
    AVLNode* T2 = x->right;
    
    x->right = y;
    y->left = T2;
    
    y->height = std::max(height(y->left), height(y->right)) + 1;
    x->height = std::max(height(x->left), height(x->right)) + 1;
    
    return x;
}

AVLNode* AVLTree::leftRotate(AVLNode* x) {
    AVLNode* y = x->right;
    AVLNode* T2 = y->left;
    
    y->left = x;
    x->right = T2;
    
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
    
    // 左右情况
    if (balance > 1 && value > node->left->value) {
        node->left = leftRotate(node->left);
        return rightRotate(node);
    }
    
    // 右左情况
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
    while (z->parent != nullptr && z->parent->isRed) {
        if (z->parent == z->parent->parent->left) {
            RBNode* y = z->parent->parent->right;
            if (y->isRed) {
                z->parent->isRed = false;
                y->isRed = false;
                z->parent->parent->isRed = true;
                z = z->parent->parent;
            } else {
                if (z == z->parent->right) {
                    z = z->parent;
                    leftRotate(z);
                }
                z->parent->isRed = false;
                z->parent->parent->isRed = true;
                rightRotate(z->parent->parent);
            }
        } else {
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
    root->isRed = false;
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

// B树实现
void BTree::insert(double value) {
    if (root == nullptr) {
        root = new BTreeNode(true);
        root->keys.push_back(value);
        return;
    }
    
    if (root->keys.size() == 2 * t - 1) {
        BTreeNode* newRoot = new BTreeNode(false);
        newRoot->children.push_back(root);
        splitChild(newRoot, 0);
        root = newRoot;
    }
    
    insertNonFull(root, value);
}

void BTree::splitChild(BTreeNode* x, int i) {
    BTreeNode* y = x->children[i];
    BTreeNode* z = new BTreeNode(y->isLeaf);
    
    for (int j = 0; j < t - 1; j++) {
        z->keys.push_back(y->keys[j + t]);
    }
    
    if (!y->isLeaf) {
        for (int j = 0; j < t; j++) {
            z->children.push_back(y->children[j + t]);
        }
    }
    
    x->keys.insert(x->keys.begin() + i, y->keys[t - 1]);
    x->children.insert(x->children.begin() + i + 1, z);
    
    y->keys.resize(t - 1);
    if (!y->isLeaf) {
        y->children.resize(t);
    }
}

void BTree::insertNonFull(BTreeNode* x, double k) {
    int i = x->keys.size() - 1;
    
    if (x->isLeaf) {
        while (i >= 0 && k < x->keys[i]) {
            i--;
        }
        x->keys.insert(x->keys.begin() + i + 1, k);
    } else {
        while (i >= 0 && k < x->keys[i]) {
            i--;
        }
        i++;
        
        if (x->children[i]->keys.size() == 2 * t - 1) {
            splitChild(x, i);
            if (k > x->keys[i]) {
                i++;
            }
        }
        insertNonFull(x->children[i], k);
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
    
    return searchInternal(x->children[i], k);
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
