#include <vector>

using Matrix = std::vector<std::vector<long long>>;

/*
 * 二维差分
 *
 * 给子矩阵 (x1, y1) 到 (x2, y2) 加 value，只修改四个角。
 * 四个标记分别表示：开始生效、向下撤销、向右撤销、右下补偿。
 * 约定使用 1-based 下标，并额外预留第 n+1 行和第 m+1 列。
 * 最后求一次二维前缀和，即可得到所有修改后的矩阵。
 * 复杂度：每次矩形修改 O(1)，最终还原 O(nm)。
 */
void add_submatrix(
    Matrix &difference,
    int x1,
    int y1,
    int x2,
    int y2,
    int value)
{
    // 左上角开始生效，下边界和右边界外撤销，右下角补回一次。
    difference[x1][y1] += value;
    difference[x2 + 1][y1] -= value;
    difference[x1][y2 + 1] -= value;
    difference[x2 + 1][y2 + 1] += value;
}

// 求二维前缀和，还原修改后的矩阵；最外层哨兵行列可忽略。
void restore_2d(Matrix &difference)
{
    int n = static_cast<int>(difference.size()) - 1;
    int m = static_cast<int>(difference[0].size()) - 1;

    for (int i = 1; i <= n; i ++)
        for (int j = 1; j <= m; j ++)
            // 当前值 = 自身标记 + 上方累计 + 左方累计 - 左上重复部分。
            difference[i][j] += difference[i - 1][j]
                                + difference[i][j - 1]
                                - difference[i - 1][j - 1];
}
