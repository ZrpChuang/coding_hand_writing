"""二叉树层序遍历（BFS）：手撕版，返回按层分组的结果。"""

# deque 两端操作都是 O(1)，比 list.pop(0) 更适合做队列。
from collections import deque


# 二叉树节点：值 + 左右孩子指针。
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# root:TreeNode，返回 [[第0层], [第1层], ...] 这样的二维列表。
def level_order(root):
    # 空树直接返回空列表，避免后续对 None 取属性。
    if root is None:
        return []
    # 结果容器：每个元素是一层的节点值列表。
    result = []
    # 队列初始只放根节点。
    queue = deque([root])
    # 队列非空说明还有未访问的层。
    while queue:
        # 当前层的节点个数：进入循环这一刻队列里全是本层节点。
        level_size = len(queue)
        # 收集本层节点值。
        level = []
        # 只弹出本层的 level_size 个节点，把它们的孩子留给下一轮。
        for _ in range(level_size):
            # 从队首取出一个节点。
            node = queue.popleft()
            # 记录该节点的值。
            level.append(node.val)
            # 左孩子存在就入队，成为下一层的一部分。
            if node.left:
                queue.append(node.left)
            # 右孩子同理。
            if node.right:
                queue.append(node.right)
        # 本层遍历完，整层追加到结果。
        result.append(level)
    return result


# 直接运行本文件时执行 demo。
if __name__ == "__main__":
    # 构造下面这棵树：
    #        3
    #       / \
    #      9  20
    #         / \
    #        15  7
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)
    # 期望输出 [[3], [9, 20], [15, 7]]。
    print("level order:", level_order(root))
