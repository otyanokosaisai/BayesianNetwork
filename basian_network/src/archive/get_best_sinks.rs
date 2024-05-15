use std::collections::HashMap;
use dashmap::DashMap;
use itertools::Itertools;
use crate::score_base::get_best_parents::get_best_parents;

pub fn get_best_sinks(v: &Vec<u8>, local_score: &DashMap<u8, DashMap<Vec<u8>, f32>>) -> HashMap<Vec<u8>, u8> {
    let mut scores: HashMap<Vec<u8>, f32> = HashMap::new();
    let mut sinks: HashMap<Vec<u8>, u8> = HashMap::new();
    let subsets = v.iter()
        .flat_map(|size| v.iter().copied().combinations(*size as usize))
        .chain(std::iter::once(v.clone()))
        .map(|combo| combo.into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for w in subsets {
        scores.insert(w.clone(), 0.0);
        sinks.insert(w.clone(), u8::MAX);  // u8::MAX を仮の "無効" 値として使用
        for (index, &sink) in w.iter().enumerate() {
            let mut upvars = w.clone();
            upvars.remove(index);  // sinkを除外
            let skore = scores.get(&upvars).unwrap() 
                + *local_score.get(&sink).unwrap().get(&get_best_parents(sink, &upvars, &local_score.get(&sink).unwrap())[&upvars]).unwrap();
            if sinks[&w] == u8::MAX || skore > *scores.get(&w).unwrap() {
                scores.insert(w.clone(), skore);
                sinks.insert(w.clone(), sink);
            }
        }
    }
    // println!("sinks: {:?}", sinks);
    // for sink in &sinks {
    //     if *sink.1 == u8::MAX {
    //         println!("sink.1 == u8::MAX, sink: {:?}", sink);
    //     }
    // }
    sinks
}
