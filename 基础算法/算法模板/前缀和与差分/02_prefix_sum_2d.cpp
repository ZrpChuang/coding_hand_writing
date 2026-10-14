#include <vector>

using Matrix = std::vector<std::vector<long long>>;

/*
 * 二维前缀和
 *
 * 约定：矩阵使用 1-based 下标，第 0 行和第 0 列补 0。
 * s[i][j] 表示左上角 (1, 1) 到右下角 (i, j) 的矩形和。
 * 查询时通过容斥：总区域 - 上方 - 左方 + 被重复减掉的左上角。
 * 复杂度：预处理 O(nm)，每次查询 O(1)，空间 O(nm)。
 */
Matrix build_prefix_sum_2d(const Matrix &a)
{
    int n = static_cast<int>(a.size()) - 1;
    int m = static_cast<int>(a[0].size()) - 1;
    Matrix s(n + 1, std::vector<long long>(m + 1));

    for (int i = 1; i <= n; i ++)
        for (int j = 1; j <= m; j ++)
            // 上方与左方相加后，左上角被计算两次，需要减去一次。
            s[i][j] = a[i][j] + s[i - 1][j] + s[i][j - 1]
                      - s[i - 1][j - 1];

    return s;
}

long long submatrix_sum(
    const Matrix &s,
    int x1,
    int y1,
    int x2,
    int y2)
{
    // 减去目标矩形上方和左方，再补回被重复减去的左上区域。
    return s[x2][y2] - s[x1 - 1][y2] - s[x2][y1 - 1]
           + s[x1 - 1][y1 - 1];
}
