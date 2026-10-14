#include <algorithm>
#include <vector>

/*
 * 离散化
 *
 * 将跨度很大但数量较少的值映射到连续下标 1..n。
 * 第一步排序并去重；第二步用 lower_bound 查询原值对应的位置。
 * 相对大小关系保持不变，适合配合数组、前缀和与树状数组使用。
 * 预处理 O(n log n)，单次查询 O(log n)。
 */
std::vector<int> discretize(std::vector<int> values)
{
    std::sort(values.begin(), values.end());
    // unique 把重复值移到尾部，并返回去重后区间的末尾。
    values.erase(
        std::unique(values.begin(), values.end()),
        values.end()
    );
    return values;
}

// lower_bound 找到第一个 >= x 的位置，再加 1 转成 1-based 下标。
int find_index(const std::vector<int> &values, int x)
{
    return std::lower_bound(values.begin(), values.end(), x)
           - values.begin() + 1;
}
