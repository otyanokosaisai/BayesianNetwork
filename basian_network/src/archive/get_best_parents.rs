use dashmap::DashMap;
use itertools::Itertools;
use std::collections::HashMap;

pub fn get_best_parents(v: u8, v_set: &Vec<u8>, local_score_v: &DashMap<Vec<u8>, f32>) -> HashMap<Vec<u8>, Vec<u8>> {
    let v_set_without_v: Vec<u8> = v_set.iter().copied().filter(|&x| x != v).collect();
    let mut best_scores = HashMap::new();
    let mut best_parent_sets = HashMap::new();
    for cs in v_set_without_v.iter().copied().powerset() {
        let cs_cloned = cs.clone();
        let score = match local_score_v.get(&cs) {
            Some(score) => score,
            None => panic!("{:?}", cs),
        };
        best_scores.insert(cs_cloned.clone(), *score.value());  // Use the cloned version in the HashMap.
        best_parent_sets.insert(cs_cloned.clone(), cs_cloned.clone());
        for j in 0..cs_cloned.len() {
            let mut cs1 = cs_cloned.clone();
            cs1.remove(j);
            let score = best_scores.get(&cs1).unwrap();
            if *score > *best_scores.get(&cs_cloned).unwrap() {
                // if cs1.len() == 0 {
                //     // println!("[] was considered and {:?}(:[]) > {:?}", *score, *best_scores.get(&cs_cloned).unwrap());
                // }
                best_scores.insert(cs_cloned.clone(), *score);
                best_parent_sets.insert(cs_cloned.clone(), best_parent_sets.get(&cs1).unwrap().clone());
            } // else if cs1.len() == 0 {
            //     println!("[] was considered but {:?}(:[]) < {:?}", *score, *best_scores.get(&cs_cloned).unwrap());
            // }
        }
    }
    best_parent_sets
}
