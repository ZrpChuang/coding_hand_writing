#include <algorithm>
#include <utility>
#include <vector>

using Interval = std::pair<int, int>;

/*
 * 区间合并
 *
 * 先按左端点排序，再从左到右维护最后一个已合并区间。
 * 若新区间与它分离，则直接加入；否则扩展最后区间的右端点。
 * 排序保证后续区间的左端点不会变小，因此只需检查最后一个区间。
 * 复杂度：排序 O(n log n)，扫描 O(n)，结果空间 O(n)。
 */
std::vector<Interval> merge_intervals(std::vector<Interval> intervals)
{
    std::vector<Interval> result;
    std::sort(intervals.begin(), intervals.end());

    for (const auto &[left, right] : intervals)
    {
        // result 为空，或当前区间与最后一个结果区间没有交集。
        if (result.empty() || result.back().second < left)
        {
            result.push_back({left, right});
        }
        else
        {
            // 存在交集，保留更大的右端点。
            result.back().second = std::max(result.back().second, right);
        }
    }

    return result;
}
