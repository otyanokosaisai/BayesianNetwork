// use dashmap::DashMap;
// use std::collections::{HashMap, HashSet};

// fn all_parent_combinations(parents: &HashSet<u8>, cft_node: &DashMap<Vec<u8>, HashMap<u8, u32>>) -> Vec<Vec<u8>> {
//     // 親ノードの値の最大値を取得し、全ての可能な組み合わせを生成
//     let max_value = *parents.iter().max().unwrap_or(&0);
//     let mut combinations = vec![];

//     // 簡単のために、ここでは親ノードが1つの場合のみを考える
//     for i in 0..=max_value {
//         for j in 0..=max_value {
//             combinations.push(vec![i, j]);
//         }
//     }
//     combinations
// }

// pub fn calculate_bic(
//     graph: &HashMap<u8, HashSet<u8>>,
//     cft: &DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u32>>>,
//     total_samples: usize
// ) -> f64 {
//     let mut total_bic = 0.0;
//     for (node, parents) in graph.iter() {
//         let cft_node = cft.get(node).unwrap();

//         if parents.is_empty() {
//             let cft_node = cft.get(node).unwrap();
//             let mut cft_sub: HashMap<u8, u32> = HashMap::new();
//             for entry in cft_node.iter() {
//                 let value_counts = entry.value();
//                 for (&k, &v) in value_counts.iter() {
//                     *cft_sub.entry(k).or_insert(0) += v;
//                 }
//             }
//             let mut log_likelihood = 0.0;
//             let total_count: u32 = cft_sub.values().sum();
//             let num_states = cft_sub.len() as usize;
//             let num_params = num_states - 1;
//             for &count in cft_sub.values() {
//                 if count > 0 {
//                     log_likelihood += (count as f64) * ((count as f64 / total_count as f64).ln());
//                 }
//             }
//             total_bic += 1.0 * log_likelihood - (num_params as f64) * (total_samples as f64).ln() / 2.0;
//             continue;
//         }

//         // 親ノードの全組み合わせを生成
//         let combinations = all_parent_combinations(parents, &cft_node);
//         let cft_sub: DashMap<Vec<u8>, HashMap<u8, u32>> = DashMap::new();

//         // 全組み合わせを処理
//         for comb in combinations {
//             let entry = cft_node.get(&comb);
//             let value_counts = match entry {
//                 Some(e) => e.value().clone(),
//                 None => HashMap::new(),  // 存在しないキーの場合は空のHashMap（カウント0）
//             };
//             cft_sub.insert(comb, value_counts);
//         }

//         // BICの計算
//         // (以前のコードを使用)
//     }
//     total_bic
// }



use dashmap::DashMap;
use std::collections::{HashMap, HashSet};

pub fn calculate_bic(
    graph: &HashMap<u8, HashSet<u8>>,
    cft: &DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u32>>>,
    total_samples: usize,
    
) -> f64 {
    let mut total_bic = 0.0;
    for (node, parents) in graph.iter() {
        if parents.len() == 0 {
            let cft_node = cft.get(node).unwrap();
            let mut cft_sub: HashMap<u8, u32> = HashMap::new();
            for entry in cft_node.iter() {
                let value_counts = entry.value();
                for (&k, &v) in value_counts.iter() {
                    *cft_sub.entry(k).or_insert(0) += v;
                }
            }
            let mut log_likelihood = 0.0;
            let total_count: u32 = cft_sub.values().sum();
            let num_states = cft_sub.len() as usize;
            let num_params = num_states - 1;
            for &count in cft_sub.values() {
                if count > 0 {
                    log_likelihood += (count as f64) * ((count as f64 / total_count as f64).ln());
                }
            }
            total_bic += 1.0 * log_likelihood - (num_params as f64) * (total_samples as f64).ln() / 2.0;
        } else {
            let cft_node = cft.get(node).unwrap();
            // println!("before parents: {:?}, node: {:?}", parents, node);
            let mut parents = parents.iter().map(|&parent| 
                if parent < *node {
                    parent
                } else {
                    parent - 1
                }
            ).collect::<Vec<u8>>();
            parents.sort();
            // println!("after parents: {:?}, node: {:?}", parents, node);
            let cft_sub: DashMap<Vec<u8>, HashMap<u8, u32>> = DashMap::new();
            for entry in cft_node.iter() {
                let new_key: Vec<u8> = entry.key().iter().enumerate().filter_map(|(i, &v)| 
                    if parents.contains(&(i as u8)) {
                        Some(v)
                    } else {
                        None
                    }
                ).collect();
                let value_counts = entry.value();
                // cft_subにnew_keyが存在しないならinsert存在するなら、HashMapの対応するキーに追加
                cft_sub.entry(new_key).and_modify(|e| {
                    for (&k, &v) in value_counts.iter() {
                        *e.entry(k).or_insert(0) += v;
                    }
                }).or_insert(value_counts.clone());
            }
            // println!("cft_sub: {:?}", cft_sub);
            let mut log_likelihood = 0.0;
            for entry in cft_sub.iter() {
                if entry.key().len() == 0 {
                    continue;
                }
                let counts = entry.value();
                let total_count: u32 = counts.values().sum();
                let num_states = counts.len() as usize;
                for &count in counts.values() {
                    if count > 0 {
                        log_likelihood += (count as f64) * ((count as f64 / total_count as f64).ln());
                    }
                }
            }

            let mut parents_values_types_nums: HashMap<u8, HashSet<u8>> = HashMap::new();
            for entry in cft_sub.iter() {
                let parent_values = entry.key();
                for (i, &parent_value) in parent_values.iter().enumerate() {
                    parents_values_types_nums.entry(i as u8).and_modify(|e| {
                        e.insert(parent_value);
                    }).or_insert({
                        let mut h = HashSet::new();
                        h.insert(parent_value);
                        h
                    });
                }
            }
            // println!("parents_values_types_nums: {:?}", parents_values_types_nums);
            let mut k = 1.0;
            for (_, parent_values_types) in parents_values_types_nums.iter() {
                k *= parent_values_types.len() as f32;
            }
            k *= cft_sub.iter().next().unwrap().value().len() as f32 - 1.0;
            let bic = 1.0 * log_likelihood - (k as f64) * (total_samples as f64).ln() / 2.0;
            total_bic += bic;
        }
    }
    total_bic
}