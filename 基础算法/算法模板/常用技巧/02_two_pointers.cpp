#include <algorithm>
#include <unordered_map>
#include <vector>

/*
 * 双指针：最长连续不重复子序列
 *
 * [j, i] 是当前滑动窗口，count 记录窗口内每个值出现的次数。
 * i 向右扩展；若 a[i] 重复，则移动 j，直到窗口重新满足无重复。
 * 因为 i、j 都只单向移动，虽然存在两层循环，总时间仍是 O(n)。
 * 空间复杂度 O(k)，k 是窗口中不同元素的数量。
 */
int longest_unique_subarray(const std::vector<int> &a)
{
    std::unordered_map<int, int> count;
    int result = 0;
    int n = static_cast<int>(a.size());

    for (int i = 0, j = 0; i < n; i ++)
    {
        // 将右端点元素加入窗口。
        count[a[i]] ++;

        // 当前元素重复时，不断收缩左端点。
        while (count[a[i]] > 1)
            count[a[j ++]] --;

        // 此时 [j, i] 一定没有重复元素。
        result = std::max(result, i - j + 1);
    }

    return result;
}
