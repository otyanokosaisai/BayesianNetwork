use std::collections::{HashMap, HashSet};
use dashmap::DashMap;

use crate::score_base::get_best_parents::get_best_parents;

pub fn ord2net(v: &Vec<u8>, ord: Vec<u8>, local_score: &DashMap<u8, DashMap<Vec<u8>, f32>>) -> HashMap<u8, HashSet<u8>> {
    let mut parents: Vec<Vec<u8>> = vec![vec![]; v.len()];
    let mut predecs: Vec<u8> = Vec::new();
    for child in (0..v.len()) {
        let ord_i = ord[child];
        predecs.sort();
        let ls = local_score.get(&ord_i).unwrap().clone();
        let bps_value = match get_best_parents(ord_i, &predecs, &ls).get(&predecs) {
            Some(bps) => bps.clone(),
            None => {
                // parents[child] = vec![];
                parents[ord_i as usize] = vec![];
                predecs.push(ord_i);
                continue;
            }
        };
        parents[ord_i as usize] = bps_value.clone(); // child -> ord_i に変更
        // parents[child] = bps_value.clone();
        predecs.push(ord_i);
        // println!("{:?} {:?}", ord_i, bps_value);
    }
    parents.into_iter().enumerate().map(|(i, p)| (i as u8, p.into_iter().collect())).collect()
}
