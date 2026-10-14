#include <vector>

/*
 * 一维差分
 *
 * difference 是原数组的差分，原数组可由它的前缀和恢复。
 * 给 [l, r] 加 value，只需标记从 l 开始生效、从 r + 1 开始失效。
 * 约定使用 1-based 下标，并额外预留 r + 1 的哨兵位置。
 * 复杂度：每次区间修改 O(1)，最终还原 O(n)。
 */
void add_range(std::vector<long long> &difference, int l, int r, int value)
{
    // l 处开始增加 value，r + 1 处撤销这次增加。
    difference[l] += value;
    difference[r + 1] -= value;
}

// 对差分数组求前缀和，即可还原所有区间修改后的值。
void restore(std::vector<long long> &difference)
{
    for (std::size_t i = 1; i < difference.size(); i ++)
        difference[i] += difference[i - 1];
}
