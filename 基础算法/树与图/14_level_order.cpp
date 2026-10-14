// 二叉树层序遍历（BFS）：手撕版，返回按层分组的结果。

#include <iostream>
#include <queue>
#include <vector>

// 二叉树节点：值 + 左右孩子指针。
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    // 构造函数：默认左右孩子为空。
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// root 是根节点，返回 {{第0层}, {第1层}, ...} 这样的二维数组。
std::vector<std::vector<int>> levelOrder(TreeNode* root) {
    // 结果容器：每个元素是一层的节点值。
    std::vector<std::vector<int>> result;
    // 空树直接返回空结果，避免对空指针取成员。
    if (root == nullptr) return result;
    // 队列初始只放根节点。
    std::queue<TreeNode*> q;
    q.push(root);
    // 队列非空说明还有未访问的层。
    while (!q.empty()) {
        // 当前层节点数：进入循环这一刻队列里全是本层节点。
        int levelSize = q.size();
        // 收集本层节点值。
        std::vector<int> level;
        // 只处理本层的 levelSize 个节点，孩子留给下一轮。
        for (int i = 0; i < levelSize; ++i) {
            // 取出队首节点并出队。
            TreeNode* node = q.front();
            q.pop();
            // 记录该节点的值。
            level.push_back(node->val);
            // 左孩子存在就入队，成为下一层的一部分。
            if (node->left) q.push(node->left);
            // 右孩子同理。
            if (node->right) q.push(node->right);
        }
        // 本层遍历完，整层追加到结果。
        result.push_back(level);
    }
    return result;
}

// 直接运行本文件时执行 demo。
int main() {
    // 构造下面这棵树：
    //        3
    //       / \
    //      9  20
    //         / \
    //        15  7
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(9);
    root->right = new TreeNode(20);
    root->right->left = new TreeNode(15);
    root->right->right = new TreeNode(7);

    // 期望输出 [[3], [9, 20], [15, 7]]。
    std::vector<std::vector<int>> ans = levelOrder(root);
    std::cout << "level order: [";
    for (size_t i = 0; i < ans.size(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < ans[i].size(); ++j) {
            std::cout << ans[i][j];
            if (j + 1 < ans[i].size()) std::cout << ", ";
        }
        std::cout << "]";
        if (i + 1 < ans.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    return 0;
}
