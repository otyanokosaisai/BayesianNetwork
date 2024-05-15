use std::collections::{HashMap, HashSet};


pub fn extract_direct_parents(edges: &HashMap<u8, HashSet<u8>>) -> HashMap<u8, HashSet<u8>> {
    // 最終結果を格納するHashMap
    let mut direct_parents = HashMap::new();

    // すべてのノードに対して直接の親を探索
    for (node, parents) in edges {
        let mut direct = HashSet::new();
        if parents.is_empty() {
            direct_parents.insert(*node, direct);
            continue;
        }
        // 他の全てのノードの親リストと比較
        for &parent in parents {
            let mut is_direct = true;
            // 他の親ノードがこの親を含んでいるかどうかを調べる
            for &other_parent in parents {
                if parent != other_parent && edges.get(&other_parent).unwrap().contains(&parent) {
                    is_direct = false;
                    break;
                }
            }
            if is_direct {
                direct.insert(parent);
            }
        }
        direct_parents.insert(*node, direct);
    }

    direct_parents
}