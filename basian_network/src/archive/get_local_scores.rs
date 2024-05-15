/*
    ct2cft : fn from ct generate cft
        ct2cft(ct, v) -> cft
*/
// vのカラムを削除して新しいcountをtupleとして(F0, F1,,)のように追加する
/*
    0111 6
    0110 15
    1010 5
    1011 7

    (focus on 4th column)
    011 (15, 6)
    101 (5,7)
*/

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use dashmap::DashMap;
use anyhow::Result;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

pub fn ct2cft(ct: Arc<DashMap<Vec<u8>, u32>>, num_valiables: u8) -> Result<DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u32>>>, Box<dyn std::error::Error>> {
    // child_index, parent_indices, parents value, child_value, count
    let all_valiables: Vec<u8> = (0..num_valiables).collect();
    let cft: DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u32>>> = DashMap::new(); // child index, parent values, child value, count

    all_valiables.par_iter().for_each(|&child_valiable_index| {
        let parent_indices: Vec<u8> = all_valiables.iter().filter(|&&v| v != child_valiable_index).map(|&v| v).collect();
        let cft_sub: DashMap<Vec<u8>, HashMap<u8, u32>> = DashMap::new();
        
        ct.iter().for_each(|entry| {
            let parent_values: Vec<u8> = parent_indices.iter()
                .map(|&i| entry.key()[i as usize]).collect();
            let child_value: u8 = entry.key()[child_valiable_index as usize];
            let count: u32 = *entry.value();
            cft_sub.entry(parent_values).and_modify(|e| {
                e.entry(child_value).and_modify(|e| {
                    *e += count;
                }).or_insert(count);
            }).or_insert({
                let mut h = HashMap::new();
                h.insert(child_value, count);
                h
            });
        });
        cft.insert(child_valiable_index, cft_sub);
    });
    Ok(cft)
}

pub fn calculate_bics_parallel(
    data: Arc<DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u32>>>>
) -> DashMap<u8, DashMap<Vec<u8>, f32>> {
    let local_scores = DashMap::new();
    let variables: Vec<u8> = data.iter().map(|entry| *entry.key()).collect();
    variables.clone().into_par_iter().enumerate().for_each(|(child_index, child)| {
        let mut parent_indices = variables.clone();
        parent_indices.remove(child_index);
        parent_indices.sort();
        let data_map = data.get(&child).unwrap().clone();
        let sample_size = calculate_sample_size(&data_map);
        let mut local_scores_for_child = DashMap::new();
        let mut visited = HashSet::new();
        calculate_bic_recursive(&mut parent_indices, &data_map, sample_size, &mut local_scores_for_child, &mut visited);
        local_scores.insert(child, local_scores_for_child);
    });
    // for entry in local_scores.iter() {
    //     for sub_entry in entry.value().iter() {
    //         println!("{:?} {:?} {:?}", entry.key(), sub_entry.key(), sub_entry.value());
    //     }
    // }
    // panic!("_");
    local_scores
}

fn calculate_bic_recursive(
    indices: &mut Vec<u8>,
    data_map: &DashMap<Vec<u8>, HashMap<u8, u32>>,
    sample_size: u32,
    local_scores_for_child: &mut DashMap<Vec<u8>, f32>,
    visited: &mut HashSet<Vec<u8>>
) {
    if visited.contains(indices) {
        return;
    }
    visited.insert(indices.to_vec());
    let bic = calculate_bic_for_parent_set(&data_map, sample_size as usize);
    local_scores_for_child.insert(indices.to_vec(), bic);
    // 各要素を一つずつ除いて新しいサブセットを生成し、再帰的に計算
    for index in 0..indices.len() {
        let mut new_indices = indices.clone();
        new_indices.remove(index);
        let new_data_for_parent_subset: DashMap<Vec<u8>, HashMap<u8, u32>> = DashMap::new();
        data_map.iter().for_each(|entry| {
            let mut key = entry.key().clone();
            let value_map = entry.value();  
            key.remove(index);
            new_data_for_parent_subset.entry(key).and_modify(|existing_map: &mut HashMap<u8, u32>| {
                for (key, value) in value_map.iter() {
                    *existing_map.entry(*key).or_insert(0) += value;
                }
            }).or_insert_with(|| value_map.clone());
        });
        calculate_bic_recursive(&mut new_indices, &new_data_for_parent_subset, sample_size, local_scores_for_child, visited);
    }
}

fn calculate_sample_size(parent_data: &DashMap<Vec<u8>, HashMap<u8, u32>>) -> u32 {
    parent_data.iter().fold(0, |acc, item| acc + item.value().values().sum::<u32>())
}

fn calculate_log_likelihood(
    data_map: &DashMap<Vec<u8>, HashMap<u8, u32>>,
) -> f32 {
    let mut total_log_likelihood = 0.0;
    for child_distribution in data_map.iter() {
        let total_count: u32 = child_distribution.value().values().sum();
        for &count in child_distribution.value().values() {
            if count > 0 {
                let probability = (count as f64) / (total_count as f64);
                total_log_likelihood += (count as f64) * probability.ln();
            }
        }
    }
    total_log_likelihood as f32
}

// fn calculate_bic_for_parent_set(parent_data: &DashMap<Vec<u8>, HashMap<u8, u32>>, sample_size: usize) -> f32 {
//     if parent_data.contains_key(&vec![]) {
//         // 親が存在しない場合の処理
//         let counts_hash = parent_data.get(&vec![]).unwrap();
//         let total_count: u32 = counts_hash.values().sum();
//         let num_states = counts_hash.len() as usize;
//         let num_params = num_states - 1;
//         let mut log_likelihood = 0.0;
//         for &count in counts_hash.values() {
//             if count > 0 {
//                 log_likelihood += (count as f32) * ((count as f32 / total_count as f32).ln());
//             }
//         }
//         return log_likelihood - (num_params as f32) * (sample_size as f32).ln() / 2.0;
//     }

//     let mut likelihood = 0.0;
//     let mut total_params = 0.0;

//     // 全ての親の組み合わせに対する子供の状態を取得
//     let child_states: HashSet<u8> = parent_data.iter()
//         .flat_map(|entry| entry.value().keys().copied().collect::<HashSet<u8>>())
//         .collect();

//     let num_child_states = child_states.len() as f32;

//     // すべての親の組み合わせとそれに対する子供の状態のカウントを考える
//     for entry in parent_data.iter() {
//         let counts = entry.value();
//         let total_count: u32 = counts.values().sum();
//         let num_states = counts.len() as f32;
//         total_params += num_states - 1.0;

//         for &count in counts.values() {
//             if count > 0 {
//                 likelihood += (count as f32) * ((count as f32 / total_count as f32).ln());
//             }
//         }
//     }

//     let k = total_params * num_child_states - 1.0; // 親の組み合わせに基づくパラメータ数
//     let bic = likelihood - k * (sample_size as f32).ln() / 2.0;

//     bic
// }

fn calculate_bic_for_parent_set(parent_data: &DashMap<Vec<u8>, HashMap<u8, u32>>, sample_size: usize) -> f32 {
    // if parent_data.into_iter().next().unwrap().key().len() == 0 {
    if parent_data.contains_key(&vec![]) {
        let counts = parent_data.into_iter().next().unwrap();
        let counts_hash = counts.value();
        // panic!("parents_data: {:?}, counts: {:?}", parent_data, counts_hash);
        let mut log_likelihood = 0.0;
        let total_count: u32 = counts_hash.values().sum();
        let num_states = counts_hash.len() as usize;
        let num_params = num_states - 1;
        for &count in counts_hash.values() {
            if count > 0 {
                log_likelihood += (count as f32) * ((count as f32 / total_count as f32).ln());
            }
        }
        return 1.0 * log_likelihood - (num_params as f32) * (sample_size as f32).ln() / 2.0;
    }

    let mut parents_values_types_nums: HashMap<u8, HashSet<u8>> = HashMap::new();
    for entry in parent_data.iter() {
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
    k *= parent_data.iter().next().unwrap().key().len() as f32 - 1.0;
    // k -= 1.0;

    let l = calculate_log_likelihood(parent_data);
    let bic = 1.0 * l - k * (sample_size as f32).ln() / 2.0;
    // let bic = 1.0 * l - k; // AIC用簡易的に
    bic
}



// /// Calculates the BIC score for a set of parent-child relationships stored in a DashMap.
// /// Assumes that sample_size is provided externally and is the total number of samples in the dataset.
// pub fn calculate_bic_for_parent_set(parent_data: &DashMap<Vec<u8>, HashMap<u8, u32>>, sample_size: usize) -> f32 {
//     let mut total_log_likelihood = 0.0;
//     let mut total_num_params = 0;

//     // Iterate over all parent sets in the data
//     for enter in parent_data.iter() {
//         let child_counts = enter.value();
//         let total_counts: u32 = child_counts.values().sum();
//         if total_counts == 0 {
//             continue; // Skip if no counts are available
//         }

//         // Calculate log likelihood for this parent set
//         let log_likelihood = child_counts.iter().fold(0.0, |acc, (&child_state, &count)| {
//             if count > 0 {
//                 acc + (count as f32) * ((count as f32 / total_counts as f32).ln())
//             } else {
//                 acc
//             }
//         });

//         // Update the total log likelihood
//         total_log_likelihood += log_likelihood;

//         // Number of parameters is the number of child states minus 1 for each parent configuration
//         let num_params = child_counts.len() - 1;
//         total_num_params += num_params;
//     }

//     // Calculate and return the BIC score
//     -2.0 * total_log_likelihood + (total_num_params as f32) * (sample_size as f32).ln()
// }

// fn calculate_bic_for_parent_set(parent_data: &DashMap<Vec<u8>, HashMap<u8, u32>>, sample_size: usize) -> f32 {
//     // if parent_data.into_iter().next().unwrap().key().len() == 0 {
//     if parent_data.contains_key(&vec![]) {
//         let counts = parent_data.into_iter().next().unwrap();
//         let counts_hash = counts.value();
//         // panic!("parents_data: {:?}, counts: {:?}", parent_data, counts_hash);
//         let mut log_likelihood = 0.0;
//         let total_count: u32 = counts_hash.values().sum();
//         let num_states = counts_hash.len() as usize;
//         let num_params = num_states - 1;
//         for &count in counts_hash.values() {
//             if count > 0 {
//                 log_likelihood += (count as f32) * ((count as f32 / total_count as f32).ln());
//             }
//         }
//         return 1.0 * log_likelihood - (num_params as f32) * (sample_size as f32).ln() / 2.0;
//     }
//     // let k = parent_data.iter().next().unwrap().key().len() as f32 - 1.0;  // パラメータ数
//     let num_params_vec = parent_data.iter().map(|entry| entry.key().len() as f32).collect::<Vec<f32>>();
//     let k = num_params_vec.iter().fold(0.0, |m, v| v.max(m)) - 1.0;
//     let l = calculate_log_likelihood(parent_data);
//     let bic = 1.0 * l - k * (sample_size as f32).ln() / 2.0;
//     // let bic = 1.0 * l - k; // AIC用簡易的に
//     bic
// }



pub fn get_local_scores(ct: DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u32>>>, score: &str) -> DashMap<u8, DashMap<Vec<u8>, f32>> {
    match score {
        "bic" => calculate_bics_parallel(Arc::new(ct)),
        // "bic2" => calculate_bics_parallel2(Arc::new(ct)),
        // "bic_staged" => calculate_bics_staged_parallel(Arc::new(ct)),
        // "aic" => calculate_aics_parallel(Arc::new(ct)),
        _ => panic!("Invalid score function. Please select 'bic' or 'aic'"),
    }
}