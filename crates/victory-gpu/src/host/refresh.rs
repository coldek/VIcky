//! Disjoint shadow refreshes, adapted from aycow's VIctory PR 1818 (GPL).

/// Most copies only request one range. Gather disjoint requests without
/// allocating on the single-range path, then merge only touching ranges.
#[derive(Clone)]
pub(super) enum RefreshRanges {
    One((usize, usize)),
    Many(Vec<(usize, usize)>),
}

impl RefreshRanges {
    pub(super) fn add(&mut self, range: (usize, usize)) {
        match self {
            Self::One((from, to)) if range.0 <= *to && *from <= range.1 => {
                *from = (*from).min(range.0);
                *to = (*to).max(range.1);
            }
            Self::One(previous) => *self = Self::Many(vec![*previous, range]),
            Self::Many(ranges) => ranges.push(range),
        }
    }

    pub(super) fn merge(&mut self) {
        if let Self::Many(ranges) = self {
            ranges.sort_unstable();
            ranges.dedup_by(|next, previous| {
                if next.0 > previous.1 {
                    return false;
                }
                previous.1 = previous.1.max(next.1);
                true
            });
            if ranges.len() == 1 {
                *self = Self::One(ranges[0]);
            }
        }
    }

    pub(super) fn slices(&self) -> &[(usize, usize)] {
        match self {
            Self::One(range) => std::slice::from_ref(range),
            Self::Many(ranges) => ranges,
        }
    }

    /// The ranges are merged before recording them as current.
    pub(super) fn covers(&self, (from, to): (usize, usize)) -> bool {
        let ranges = self.slices();
        let end = ranges.partition_point(|&(start, _)| start <= from);
        end != 0 && to <= ranges[end - 1].1
    }
}
