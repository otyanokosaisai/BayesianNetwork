use std::collections::HashSet;
use std::collections::VecDeque;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use std::thread;
use chrono::Local;
use csv::ReaderBuilder;
use dashmap::DashMap;
use dashmap::DashSet;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use std::error::Error;
use anyhow::Result;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rand::prelude::SliceRandom;
use std::fmt::Write;
use statrs::function::gamma::ln_gamma;
use crate::common::core::utils::{Setting, read_settings_from_file, read_dot_file};

fn build_network_from_graph(graph: DiGraph<u8, ()>) -> Network {
    let mut network_values: HashMap<u8, Vec<u8>> = HashMap::new();

    for edge in graph.edge_references() {
        let source = edge.source().index() as u8;
        let target = edge.target().index() as u8;
        let empty_node = Vec::new();
        
        network_values
            .entry(source)
            .or_insert(empty_node.clone())
            .push(target);
    }
    // エッジにおいて存在しないノードを親ナシノードとして追加   
    for node in graph.node_indices() {
        let node = node.index() as u8;
        if !network_values.contains_key(&node) {
            network_values.insert(node, Vec::new());
        }
    }

    Network { 
        network_values: network_values.clone(),
    }
}

pub fn load_data(setting_path: &str) -> Result<DataContainer, Box<dyn Error>> {
    let setting = read_settings_from_file(setting_path)?;
    load_data_from_setting(setting)
}

pub fn load_data_from_setting(setting: Setting) -> Result<DataContainer, Box<dyn Error>> {    
    let file = File::open(&setting.data_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let headers: Vec<String> = rdr.headers()?.iter().map(|s| s.to_string()).collect();
    let mut data = Vec::new();
    let mut category_maps: Vec<HashMap<String, u8>> = vec![HashMap::new(); headers.len()];
    for result in rdr.records() {
        let record = result?;
        let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        data.push(row.iter().enumerate().map(|(i, value)| {
            let map = &mut category_maps[i];
            let len = map.len();
            *map.entry(value.clone()).or_insert_with(|| len as u8)
        }).collect::<Vec<u8>>());
    }

    let train_size = (data.len() as f64 * (1.0 - setting.valid_rate)).round() as usize;
    let (train_data, valid_data) = data.split_at(train_size);

    let train_data = Arc::new(train_data.to_vec());
    let valid_data = Arc::new(valid_data.to_vec());

    let train_freq_map = Arc::new(DashMap::new());
    let valid_freq_map = Arc::new(DashMap::new());

    let num_cpus = num_cpus::get();
    let train_chunk_size = (train_data.len() + num_cpus - 1) / num_cpus;
    let valid_chunk_size = (valid_data.len() + num_cpus - 1) / num_cpus;

    let train_handles: Vec<_> = (0..num_cpus).map(|i| {
        let train_data = Arc::clone(&train_data);
        let train_freq_map = Arc::clone(&train_freq_map);
        thread::spawn(move || {
            let start = i * train_chunk_size;
            let end = std::cmp::min((i + 1) * train_chunk_size, train_data.len());
            for row in &train_data[start..end] {
                train_freq_map.entry(row.clone()).and_modify(|e| *e += 1).or_insert(1);
            }
        })
    }).collect();

    let valid_handles: Vec<_> = (0..num_cpus).map(|i| {
        let valid_data = Arc::clone(&valid_data);
        let valid_freq_map = Arc::clone(&valid_freq_map);
        thread::spawn(move || {
            let start = i * valid_chunk_size;
            let end = std::cmp::min((i + 1) * valid_chunk_size, valid_data.len());
            for row in &valid_data[start..end] {
                valid_freq_map.entry(row.clone()).and_modify(|e| *e += 1).or_insert(1);
            }
        })
    }).collect();

    for handle in train_handles {
        handle.join().unwrap();
    }

    for handle in valid_handles {
        handle.join().unwrap();
    }

    let headers_index_map: HashMap<u8, String> = headers.iter().enumerate().map(|(i, h)| (i as u8, h.to_string())).collect();
    let final_category_maps: HashMap<u8, HashMap<String, u8>> = category_maps.into_iter().enumerate()
        .map(|(i, map)| (i as u8, map))
        .collect();
    let train_sample_size = train_freq_map.iter().map(|entry| *entry.value()).sum();
    let valid_sample_size = valid_freq_map.iter().map(|entry| *entry.value()).sum();

    let compare_network = if Path::new(&setting.compare_network_path).exists() {
        let headers_tmp: HashMap<&str, u8> = headers_index_map.iter().map(|(k, v)| (v.as_str(), *k)).collect();
        let dot_graph = read_dot_file(&setting.compare_network_path, &headers_tmp)?;
        build_network_from_graph(dot_graph)
    } else {
        Network {
            network_values: HashMap::new(),
        }
    };

    let cross_table = CrossTable {
        ct_values: train_freq_map.as_ref().clone(),
    };
    let cross_table_valid = CrossTable {
        ct_values: valid_freq_map.as_ref().clone(),
    };

    let cft = cross_table.ct2cft();
    let cft_valid = cross_table_valid.ct2cft();

    Ok(DataContainer {
        setting: setting.clone(),
        cft,
        cft_valid,
        category_maps: final_category_maps,
        header: headers_index_map,
        sample_size: train_sample_size,
        sample_size_valid: valid_sample_size,
        network: Network {
            network_values: HashMap::new(),
        },
        scoring_method: ScoringMethod {
            method: ScoringMethodName::from_string(&setting.method, setting.bdeu_ess),
        },
        compare_network,
    })
}

#[derive(Debug)]
pub struct DataContainer {
    pub setting: Setting,
    pub cft: CrossFrequencyTable,
    pub cft_valid: CrossFrequencyTable,
    pub category_maps: HashMap<u8, HashMap<String, u8>>,
    pub header: HashMap<u8, String>,
    pub sample_size: u64,
    pub sample_size_valid: u64,
    pub network: Network,
    pub scoring_method: ScoringMethod,
    pub compare_network: Network,
}

impl DataContainer {
    pub fn learn(&mut self) {
        let ls = self.cft.get_local_scores(self.sample_size, &self.scoring_method, &self.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect());
        let mut tmp_network = ls.construct_initial_bayesian_network();
        tmp_network.construct_scc_network();
        tmp_network.construct_order();
        tmp_network.order_based_search(&ls);
        self.network = Network {
            network_values: tmp_network.bayesian_network.clone(),
        };
    }
    pub fn evaluate(&self) -> f64 {
        let category_nums: HashMap<u8, u8> = self.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect();
        let score = self.network.evaluate(&self.cft_valid, self.sample_size as usize, &self.scoring_method, &category_nums);
        score
    }
    pub fn compare(&mut self) {
        let optimized_score = self.evaluate();
        let category_nums: HashMap<u8, u8> = self.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect();
        let score = self.compare_network.evaluate(&self.cft_valid, self.sample_size_valid as usize, &self.scoring_method, &category_nums);
        println!("[{:?}] optimized_score: {:?} (ans: {:?})", self.scoring_method.method, optimized_score, score);
        let cpdag1 = self.network.to_cpdag();
        let cpdag2 = self.compare_network.to_cpdag();
        let hamming_distance = cpdag1.hamming_distance(&cpdag2);
        println!("[Hamming distance(CPDAG)]: {:?}", hamming_distance);
    }
    pub fn compare_all(&mut self) -> Vec<String> {
        let mut output: Vec<String> = Vec::new();

        self.scoring_method.method = ScoringMethodName::Aic;
        let optimized_score = self.evaluate();
        let category_nums: HashMap<u8, u8> = self.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect();
        let score = self.compare_network.evaluate(&self.cft_valid, self.sample_size_valid as usize, &self.scoring_method, &category_nums);
        output.push(format!("[Aic] optimized_score: {:?}, ans_score: {:?}", optimized_score, score));

        self.scoring_method.method = ScoringMethodName::Bic;
        let optimized_score = self.evaluate();
        let score = self.compare_network.evaluate(&self.cft_valid, self.sample_size_valid as usize, &self.scoring_method, &category_nums);
        output.push(format!("[Bic] optimized_score: {:?}, ans_score: {:?}", optimized_score, score));

        self.scoring_method.method = ScoringMethodName::BDeu(self.setting.bdeu_ess);
        let optimized_score = self.evaluate();
        let score = self.compare_network.evaluate(&self.cft_valid, self.sample_size_valid as usize, &self.scoring_method, &category_nums);
        output.push(format!("[BDeu({:?})] optimized_score: {:?}, ans_score: {:?}", self.setting.bdeu_ess, optimized_score, score));
    
        let cpdag1 = self.network.to_cpdag();
        let cpdag2 = self.compare_network.to_cpdag();
        let hamming_distance = cpdag1.hamming_distance(&cpdag2);
        output.push(format!("[hamming_distance]: {:?}", hamming_distance));
    
        // self.network.cpdag = cpdag1;
        // self.compare_network.cpdag = cpdag2;
    
        output
    }
    pub fn visualize(&self) {
        match self.network.render_graph_to_dot(&self.header, &self.setting.saving_dir) {
            Ok(_) => {
            },
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}

#[derive(Debug)]
pub struct CrossTable {
    pub ct_values: DashMap<Vec<u8>, u64>,
}

#[derive(Debug)]
pub struct CrossFrequencyTable {
    pub cft_values: DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u64>>>,
}
#[derive(Debug)]
pub struct LocalScores {
    pub ls_values: DashMap<u8, DashMap<Vec<u8>, f64>>,
}
#[derive(Debug)]
pub struct TmpNetwork {
    pub network_values: HashMap<u8, Vec<u8>>,
    pub scc_network_values: HashMap<u8, Vec<u8>>, // scc_group, nodes
    pub scc_network : HashMap<u8, Vec<u8>>, // scc_group, to_scc_group
    pub orders: Vec<u8>, 
    pub bayesian_network:HashMap<u8, Vec<u8>>,
    pub cpdag: CPDAG,
}

#[derive(Debug)]
pub struct Network {
    pub network_values: HashMap<u8, Vec<u8>>,
}

#[derive(Debug)]
pub enum ScoringMethodName {
    Bic,
    Aic,
    BDeu(f64),
}

impl ScoringMethodName {
    pub fn from_string(method: &str, bdeu_ess: f64) -> ScoringMethodName {
        match method {
            "bic" => ScoringMethodName::Bic,
            "aic" => ScoringMethodName::Aic,
            "bdeu" => ScoringMethodName::BDeu(bdeu_ess),
            _ => panic!("method must be 'bic' or 'aic' or 'bdeu'"),
        }
    }
}

#[derive(Debug)]
pub struct ScoringMethod {
    pub method: ScoringMethodName,
}

impl ScoringMethod {
    fn calculate_recursive(
        &self,
        child: u8,
        indices: &Vec<u8>,
        remains: &Vec<u8>,
        data_map: &DashMap<Vec<u8>, HashMap<u8, u64>>,
        sample_size: u64,
        local_scores_for_child: &DashMap<Vec<u8>, f64>,
        visited: &DashSet<Vec<u8>>,
        drop_out: &DashSet<Vec<u8>>,
        category_map: &HashMap<u8, u8>,
        previous_score: Option<f64>,
    ) {
        if visited.contains(indices) || drop_out.iter().any(|e| e.iter().all(|&i| indices.contains(&i))) {
            return;
        }
        visited.insert(indices.to_vec());
        let new_data_for_parent_subset: DashMap<Vec<u8>, HashMap<u8, u64>> = DashMap::new();
        data_map.iter().for_each(|entry| {
            let key = indices.iter().map(|&i| {
                if i < child {
                    *entry.key().get(i as usize).unwrap()
                } else {
                    *entry.key().get(i as usize - 1).unwrap()
                }
            }).collect();
            let value_map = entry.value();
            new_data_for_parent_subset.entry(key).and_modify(|existing_map: &mut HashMap<u8, u64>| {
                for (key, value) in value_map.iter() {
                    *existing_map.entry(*key).or_insert(0) += value;
                }
            }).or_insert_with(|| value_map.clone());
        });
        let local_score = self.calculate_for_parent_set(&new_data_for_parent_subset, sample_size as usize, indices, child, category_map);
        if let Some(previous_score) = previous_score {
            // if local_score < previous_score { // socreが下がったら探索を打ち切り、drop_outに追加
            if local_score < previous_score { // && indices.len() > 1 { // socreが下がったら探索を打ち切り、drop_outに追加 空が取り除かれるべきなのかは要チェック
                // println!("drop_out: {:?} {:?}",child, indices);
                drop_out.insert(indices.clone());
                return;
            }
        }
        local_scores_for_child.insert(indices.clone(), local_score);
        remains.par_iter().enumerate().for_each(|(i, &index)| {
            let mut new_indices = indices.clone();
            new_indices.push(index);
            new_indices.sort();
            let mut new_remains = remains.clone();
            new_remains.remove(i);
            self.calculate_recursive(child, &new_indices, &new_remains, data_map, sample_size, local_scores_for_child, visited, drop_out, category_map, Some(local_score));
        });
    }
        
    fn calculate_for_parent_set(&self, parent_data: &DashMap<Vec<u8>, HashMap<u8, u64>>, sample_size: usize, parents: &Vec<u8>, child: u8, category_map: &HashMap<u8, u8>) -> f64 {
        let k: f64 = parents.iter().map(|&i| category_map[&i] as f64).product::<f64>() * (category_map[&child] as f64 - 1.0);
        let l: f64 = ScoringMethod::calculate_log_likelihood(parent_data);
        match self.method {
            ScoringMethodName::Bic => l - k * (sample_size as f64).ln() / 2.0,
            ScoringMethodName::Aic => l - k,
            ScoringMethodName::BDeu(ess) => ScoringMethod::calculate_bdeu_score(parent_data, parents, child, category_map, ess),
        }
    }
    fn calculate_log_likelihood(
        data_map: &DashMap<Vec<u8>, HashMap<u8, u64>>,
    ) -> f64 {
        data_map.iter().map(|entry| {
            let counts = entry.value();
            let total_count: u64 = counts.values().sum();
            counts.values().filter(|&&count| count > 0).map(|&count| {
                (count as f64) * ((count as f64 / total_count as f64).ln())
            }).sum::<f64>()
        }).sum()
    }
    // BDeuスコアの計算関数
    fn calculate_bdeu_score(parent_data: &DashMap<Vec<u8>, HashMap<u8, u64>>, parents: &Vec<u8>, child: u8, category_map: &HashMap<u8, u8>, ess: f64) -> f64 {
        let q_i = parents.iter().map(|&i| category_map[&i] as f64).product::<f64>();
        let r_i = category_map[&child] as f64;
        let alpha_ij = ess / q_i; // 親の状態に分配されるα
        let alpha_ijk = ess / (q_i * r_i); // 親と子の状態に分配されるα

        let mut score = 0.0;

        for parent_state in parent_data.iter() {
            let parent_state_counts = parent_state.value();
            let n_ij = parent_state_counts.values().sum::<u64>() as f64; // 親の状態の総数

            score += ln_gamma(alpha_ij) - ln_gamma(alpha_ij + n_ij);

            for &count in parent_state_counts.values() {
                score += ln_gamma(alpha_ijk + count as f64) - ln_gamma(alpha_ijk);
            }
        }

        if parent_data.len() == 0 {
            // println!("parent_data is empty");
            score += ln_gamma(alpha_ij) - ln_gamma(alpha_ij + 0.0);
            score += ln_gamma(alpha_ijk) - ln_gamma(alpha_ijk);
        }

        score
    }
}

impl CrossTable {
    pub fn ct2cft(&self) -> CrossFrequencyTable {
        let num_valiables = self.ct_values.iter().next().unwrap().key().len() as u8;
        let all_valiables: Vec<u8> = (0..num_valiables).collect();
        let cft: DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u64>>> = DashMap::new(); // child index, parent values, child value, count
        all_valiables.par_iter().for_each(|&child_valiable_index| {
            let parent_indices: Vec<u8> = all_valiables.iter().filter(|&&v| v != child_valiable_index).map(|&v| v).collect();

            let cft_sub: DashMap<Vec<u8>, HashMap<u8, u64>> = DashMap::new();
            self.ct_values.iter().for_each(|entry| {
                let parent_values: Vec<u8> = parent_indices.iter()
                    .map(|&i| entry.key()[i as usize]).collect();
                let child_value: u8 = entry.key()[child_valiable_index as usize];
                let count: u64 = *entry.value();
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
        CrossFrequencyTable {
            cft_values: cft,
        }
    }
}

impl CrossFrequencyTable {
    pub fn get_local_scores(&self, sample_size: u64, scoring_method: &ScoringMethod, category_map: &HashMap<u8, u8>) -> LocalScores {
        LocalScores {
            ls_values: self.bic(sample_size, scoring_method, category_map),
        }
    }
    pub fn bic(&self, sample_size: u64, scoring_method: &ScoringMethod, category_map: &HashMap<u8, u8>) -> DashMap<u8, DashMap<Vec<u8>, f64>> {
        let data = &self.cft_values;
        let local_scores = DashMap::new();
        let variables: Vec<u8> = data.iter().map(|entry| *entry.key()).collect();
        variables.clone().into_par_iter().for_each(|child| {
            let parent_indices: Vec<u8> = vec![];
            let remains: Vec<u8> = variables.clone().into_iter().filter(|&i| i != child).collect();
            let data_map = data.get(&child).unwrap().clone();
            let mut local_scores_for_child = DashMap::new();
            let mut visited: DashSet<Vec<u8>> = DashSet::new();
            let mut drop_out: DashSet<Vec<u8>> = DashSet::new(); // 親集合として考慮しなくていい組合わせ
            scoring_method.calculate_recursive(child, &parent_indices, &remains, &data_map, sample_size, &mut local_scores_for_child, &mut visited,&mut drop_out, category_map, None);
            drop_out.iter().for_each(|e| {
                let parent_sets = local_scores_for_child.iter().map(|entry| entry.key().clone()).collect::<Vec<Vec<u8>>>();
                parent_sets.iter().for_each(|parent_set| {
                    if e.iter().all(|i| parent_set.contains(i)) {
                        local_scores_for_child.remove(parent_set);
                    }
                });
            });
            local_scores.insert(child, local_scores_for_child);
        });
        local_scores
    }
}

impl LocalScores {
    pub fn construct_initial_bayesian_network(&self) -> TmpNetwork {
        let mut best_parents: HashMap<u8, Vec<u8>> = HashMap::new();
        for entry in self.ls_values.iter() {
            // スコアが最大のものを取り出す
            let mut score = f64::MIN;  // scoreの初期化をループの中に移動
            let mut best_parent = Vec::new();
            for sub_entry in entry.value().iter() {
                if *sub_entry.value() > score {
                    score = *sub_entry.value();  // 最大値を更新
                    best_parent = sub_entry.key().clone();
                }
            }
            best_parents.insert(*entry.key(), best_parent);
        }
        // println!("{:?}", best_parents);
        TmpNetwork {
            network_values: best_parents,
            scc_network_values: HashMap::new(),
            scc_network: HashMap::new(),
            orders: Vec::new(),
            bayesian_network: HashMap::new(),
            cpdag: CPDAG::default(),
        }
    }
}


impl TmpNetwork {
    pub fn construct_scc_network(&mut self) {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices = HashMap::new();
        let mut lowlink = HashMap::new();
        let mut in_stack = HashMap::new();
        let mut scc_index = 0;
        let mut node_to_scc = HashMap::new();
        let nodes = self.network_values.keys().cloned().collect::<Vec<u8>>();

        for node in nodes {
            if !indices.contains_key(&node) {
                self.strongconnect(node, &mut index, &mut stack, &mut indices, &mut lowlink, &mut in_stack, &mut scc_index, &mut node_to_scc);
            }
        }

        // SCC間の接続を設定する
        let mut scc_connections: HashMap<u8, HashSet<u8>> = HashMap::new();
        for (node, connections) in &self.network_values {
            // println!("[debug] node: {:?}, connections: {:?}", node, connections);
            let from_scc = node_to_scc[node];
            for &connected_node in connections {
                let to_scc = node_to_scc[&connected_node];
                if from_scc != to_scc {
                    scc_connections.entry(from_scc).or_insert_with(HashSet::new).insert(to_scc);
                }
            }
        }
        for scc_idx in 0..scc_index {
            if scc_connections.get(&scc_idx).is_none() {
                scc_connections.insert(scc_idx, HashSet::new());
            }
        }

        // HashSetからVecへ変換
        self.scc_network.clear();
        for (scc, connections) in scc_connections {
            self.scc_network.insert(scc, connections.into_iter().collect());
        }
    }

    fn strongconnect(
        &mut self,
        v: u8,
        index: &mut usize,
        stack: &mut Vec<u8>,
        indices: &mut HashMap<u8, usize>,
        lowlink: &mut HashMap<u8, usize>,
        in_stack: &mut HashMap<u8, bool>,
        scc_index: &mut u8,
        node_to_scc: &mut HashMap<u8, u8>,
    ) {
        indices.insert(v, *index);
        lowlink.insert(v, *index);
        *index += 1;
        stack.push(v);
        in_stack.insert(v, true);

        // Consider successors of v
        let successors = self.network_values.get(&v).unwrap().clone();
        for w in successors {
            if !indices.contains_key(&w) {
                self.strongconnect(w, index, stack, indices, lowlink, in_stack, scc_index, node_to_scc);
                lowlink.insert(v, std::cmp::min(*lowlink.get(&v).unwrap(), *lowlink.get(&w).unwrap()));
            } else if *in_stack.get(&w).unwrap_or(&false) {
                lowlink.insert(v, std::cmp::min(*lowlink.get(&v).unwrap(), *indices.get(&w).unwrap()));
            }
        }

        if let Some(low) = lowlink.get(&v) {
            if *low == *indices.get(&v).unwrap() {
                let mut scc = Vec::new();
                while let Some(w) = stack.pop() {
                    in_stack.insert(w, false);
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                self.scc_network_values.insert(*scc_index, scc.clone());
                for &node in &scc {
                    node_to_scc.insert(node, *scc_index);
                    // ノードに対応するエントリがなければ空のベクトルを挿入
                    self.network_values.entry(node).or_insert(vec![]);
                }
                *scc_index += 1;
            }
        }

    }

    // 各SCCのベイジアンネットワークを計算
    pub fn construct_order(&mut self) {
        let mut orders: Vec<u8> = vec![];
        for oscc in TmpNetwork::topological_sort(&self.scc_network).unwrap() {
            let scc = self.scc_network_values.get(&oscc).unwrap();
            if scc.len() > 1 {
                // random order
                let mut rng = rand::thread_rng();
                let mut order: Vec<u8> = scc.clone();
                order.shuffle(&mut rng);
                orders.extend(order);
            } else if scc.len() == 1 {
                orders.push(scc[0]);
            } else {
                panic!("scc does not contain nodes: {:?}", oscc);
            }
        }
        self.orders = orders.clone();
    }

    // トポロジカルソートの関数
    pub fn topological_sort(network: &HashMap<u8, Vec<u8>>) -> Option<Vec<u8>> {
        let mut in_degree = HashMap::new();
        let mut zero_in_degree_queue = VecDeque::new();
        let mut sorted_list = Vec::new();

        // 全ノードの入次数を初期化
        for node in network.keys() {
            in_degree.entry(*node).or_insert(0);
        }

        // 各ノードの入次数を計算
        for edges in network.values() {
            for &edge in edges {
                *in_degree.entry(edge).or_insert(0) += 1;
            }
        }

        // 入次数が0のノードをキューに追加
        for (&node, &degree) in &in_degree {
            if degree == 0 {
                zero_in_degree_queue.push_back(node);
            }
        }

        // トポロジカルソートの実行
        while let Some(node) = zero_in_degree_queue.pop_front() {
            sorted_list.push(node);
            if let Some(edges) = network.get(&node) {
                for &edge in edges {
                    let degree = in_degree.get_mut(&edge).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        zero_in_degree_queue.push_back(edge);
                    }
                }
            }
        }

        // すべてのノードがソートされたか確認（サイクルの検出）
        if sorted_list.len() == network.len() {
            Some(sorted_list)
        } else {
            // println!("{:?}", sorted_list);
            None // サイクルが存在する場合、Noneを返す
        }
    }
    pub fn order_based_search(&mut self, local_scores: &LocalScores) {
        self.bayesian_network.clear();
        let mut invalid_nodes: HashSet<u8> = HashSet::new();
        for node in &self.orders {
            // println!("node: {:?} begin, invlids: {:?}", node, invalid_nodes);
            let mut best_score = f64::MIN;
            let local_score_node: dashmap::mapref::one::Ref<u8, DashMap<Vec<u8>, f64>, std::hash::RandomState> = local_scores.ls_values.get(node).unwrap();
            let mut best_parents = Vec::new();
            for entry in local_score_node.iter() {
                let parent_candidates = entry.key();
                if invalid_nodes.iter().any(|i| parent_candidates.contains(i)) {
                    // println!("  Skipping parent candidates: {:?} due to invalid node", parent_candidates);
                    continue;
                }
                let score = *entry.value();
                if score > best_score {
                    best_score = score;
                    best_parents = parent_candidates.clone();
                }   
            }
            // println!("{:?} {:?}", node, best_parents);
            self.bayesian_network.insert(*node, best_parents);
            invalid_nodes.insert(*node);
        }
    }
}

impl Network {
    pub fn evaluate(
        &self,
        cft: &CrossFrequencyTable,
        total_samples: usize,
        scoreing_method: &ScoringMethod,
        category_map: &HashMap<u8, u8>,
    ) -> f64 {
        let graph = &self.network_values;
        let cft = &cft.cft_values;
        let mut total_bic = 0.0;
        for (node, parents) in graph.iter() {

            let cft_sub: DashMap<Vec<u8>, HashMap<u8, u64>> = DashMap::new();
            let mut parents  = parents.iter().map(|&parent| 
                if parent < *node {
                    parent
                } else {
                    parent - 1
                }
            ).collect::<Vec<u8>>();
            parents.sort();
            for entry in cft.get(node).unwrap().iter() {
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
            let bic = scoreing_method.calculate_for_parent_set(&cft_sub, total_samples, &parents, *node, &category_map) as f64;
            total_bic += bic;
        }
        total_bic
    }
    pub fn render_graph_to_dot(
        &self,
        headers_index_map: &HashMap<u8, String>,
        save_dir: &str,
    ) -> Result<(), std::io::Error> {
        let graph = &self.network_values;
        let mut dot_output = String::from("digraph G {\n");
        let unknown = "?".to_string();
        for (source, targets) in graph {
            let source_name = headers_index_map.get(source).unwrap_or(&unknown);
            for target in targets {
                let target_name = headers_index_map.get(target).unwrap_or(&unknown);
                writeln!(&mut dot_output, "    \"{}\" -> \"{}\";", target_name, source_name).unwrap();
            }
        }
        dot_output.push_str("}\n");
        let now = Local::now();
        let filename = format!("{}/bn_{}", save_dir, now.format("%Y%m%d%H%M%S"));
        fs::write(&format!("{}.dot", filename), dot_output)?;
        println!("Run 'dot -Tpng {}.dot > {}.png' to generate a PNG file", filename, filename);
        Ok(())
    }
    pub fn to_cpdag(&self) -> CPDAG {
        let network: HashMap<u8, Vec<u8>> = self.network_values.iter().map(|(k, v)| (*k, v.iter().map(|&v| v).collect())).collect();
        create_cpdag(network)
    }
}


#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Node {
    id: u8,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Edge {
    from: Node,
    to: Node,
}

#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct CPDAG {
    nodes: Vec<Node>,
    directed_edges: HashSet<Edge>,
    undirected_edges: HashSet<Edge>,
    v_structures: HashSet<Edge>,
}

impl CPDAG {
    fn new(nodes: Vec<Node>) -> Self {
        CPDAG {
            nodes,
            directed_edges: HashSet::new(),
            undirected_edges: HashSet::new(),
            v_structures: HashSet::new(),
        }
    }

    fn add_directed_edge(&mut self, edge: Edge) {
        self.directed_edges.insert(edge);
    }

    fn add_undirected_edge(&mut self, edge: Edge) {
        self.undirected_edges.insert(edge);
    }

    fn mark_as_v_structure(&mut self, edge: Edge) {
        self.v_structures.insert(edge);
    }

    fn is_marked_as_v_structure(&self, edge: &Edge) -> bool {
        self.v_structures.contains(edge)
    }
    fn hamming_distance(&self, other: &CPDAG) -> u64 {
        let mut distance = 0;
        let mut all_edges = self.directed_edges.union(&self.undirected_edges).cloned().collect::<HashSet<_>>();
        all_edges = all_edges.union(&other.directed_edges).cloned().collect();
        all_edges = all_edges.union(&other.undirected_edges).cloned().collect();
        for edge in all_edges {
            let self_has_directed = self.directed_edges.contains(&edge);
            let other_has_directed = other.directed_edges.contains(&edge);
            let reverse_edge = Edge { from: edge.to.clone(), to: edge.from.clone() };
            let self_has_undirected = self.undirected_edges.contains(&edge) || self.undirected_edges.contains(&reverse_edge);
            let other_has_undirected = other.undirected_edges.contains(&edge) || other.undirected_edges.contains(&reverse_edge);
            if self_has_directed != other_has_directed || self_has_undirected != other_has_undirected {
                distance += 1;
            }
        }
        distance
    }
    #[allow(dead_code)]
    pub fn render_graph_to_dot(
        &self,
        headers_index_map: &HashMap<u8, String>,
        save_dir: &str,
    ) -> Result<(), std::io::Error> {
        let mut dot_output = String::from("digraph G {\n");
        let unknown = "?".to_string();

        for edge in &self.directed_edges {
            let source_name = headers_index_map.get(&edge.from.id).unwrap_or(&unknown);
            let target_name = headers_index_map.get(&edge.to.id).unwrap_or(&unknown);
            writeln!(&mut dot_output, "    \"{}\" -> \"{}\";", source_name, target_name).unwrap();
        }

        for edge in &self.undirected_edges {
            let source_name = headers_index_map.get(&edge.from.id).unwrap_or(&unknown);
            let target_name = headers_index_map.get(&edge.to.id).unwrap_or(&unknown);
            writeln!(&mut dot_output, "    \"{}\" -> \"{}\" [dir = none];", source_name, target_name).unwrap();
        }

        dot_output.push_str("}\n");
        let now = Local::now();
        // let filename = format!("./figures/cpdag_graph{}.dot", now.format("%Y%m%d%H%M%S"));
        // fs::create_dir_all("./figures")?;
        let filename = format!("{}/cpdag_{}", save_dir, now.format("%Y%m%d%H%M%S"));
        fs::write(&format!("{}.dot", filename), dot_output)?;
        println!("Run 'dot -Tpng {}.dot > {}.png' to generate a PNG file", filename, filename);
        Ok(())
    }
}

pub fn create_cpdag(network: HashMap<u8, Vec<u8>>) -> CPDAG {
    let nodes: Vec<Node> = network.keys().map(|&id| Node { id }).collect();
    let edges: Vec<Edge> = network.iter().flat_map(|(to, froms)| {
        froms.iter().map(move |&from| Edge { from: Node { id: from }, to: Node { id: *to } })
    }).collect();
    _create_cpdag(nodes, edges)
}



fn _create_cpdag(nodes: Vec<Node>, edges: Vec<Edge>) -> CPDAG {
    let mut cpdag = CPDAG::new(nodes);

    // ノードごとに入次数のエッジを収集
    let mut incoming_edges: HashMap<&Node, Vec<&Edge>> = HashMap::new();
    let mut outgoing_edges: HashMap<&Node, Vec<&Edge>> = HashMap::new();

    for edge in &edges {
        incoming_edges.entry(&edge.to).or_default().push(edge);
        outgoing_edges.entry(&edge.from).or_default().push(edge);
    }
    // V構造の特定
    for (&node, in_edges) in &incoming_edges {
        if in_edges.len() > 1 {
            for i in 0..in_edges.len() {
                for j in (i + 1)..in_edges.len() {
                    let edge1 = in_edges[i];
                    let edge2 = in_edges[j];
                    let parent1 = &edge1.from;
                    let parent2 = &edge2.from;

                    // 両親が直接的な関係を持たない場合、V構造
                    if !outgoing_edges.get(&parent1).map_or(false, |edges| edges.iter().any(|e| e.to == *parent2)) &&
                       !outgoing_edges.get(&parent2).map_or(false, |edges| edges.iter().any(|e| e.to == *parent1)) {
                        cpdag.mark_as_v_structure(edge1.clone());
                        cpdag.mark_as_v_structure(edge2.clone());

                        // 子ノードから出るエッジもすべて有向エッジとしてマーク
                        if let Some(child_edges) = outgoing_edges.get(&node) {
                            for &child_edge in child_edges {
                                cpdag.mark_as_v_structure(child_edge.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    // V構造に関連するエッジを無向エッジに変換する
    for (_, in_edges) in &incoming_edges {
        for &edge in in_edges {
            if cpdag.is_marked_as_v_structure(&edge) {
                cpdag.add_directed_edge(edge.clone());
            } else {
                cpdag.add_undirected_edge(edge.clone());
            }
        }
    }

    cpdag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance_4_nodes() {
        let network1: HashMap<u8, Vec<u8>> = vec![
            (2, vec![1]),   // 1 -> 2
            (3, vec![1, 2]),// 1 -> 3, 2 -> 3
            (4, vec![3]),   // 3 -> 4
        ].into_iter().collect();

        let network2: HashMap<u8, Vec<u8>> = vec![
            (2, vec![1]),   // 1 -> 2
            (3, vec![2]),   // 2 -> 3
            (4, vec![1, 3]),// 1 -> 4, 3 -> 4
        ].into_iter().collect();

        let cpdag1: CPDAG = create_cpdag(network1);
        let cpdag2 = create_cpdag(network2);
        println!("{:?}", cpdag1);
        println!("{:?}", cpdag2);
        let distance = cpdag1.hamming_distance(&cpdag2);
        assert_eq!(distance, 3); // Edges differ
        // assert_eq!(1,2);
    }
    #[test]
    fn test_hamming_distance_5_nodes() {
        let network1: HashMap<u8, Vec<u8>> = vec![
            (2, vec![1]),   // 1 -> 2
            (3, vec![1, 2]),// 1 -> 3, 2 -> 3
            (4, vec![3]),   // 3 -> 4
            (5, vec![4]),   // 4 -> 5
        ].into_iter().collect();

        let network2: HashMap<u8, Vec<u8>> = vec![
            (2, vec![1]),   // 1 -> 2
            (3, vec![2]),   // 2 -> 3
            (4, vec![1, 3]),// 1 -> 4, 3 -> 4
            (5, vec![1]),   // 1 -> 5
        ].into_iter().collect();

        let cpdag1 = create_cpdag(network1);
        let cpdag2 = create_cpdag(network2);

        let distance = cpdag1.hamming_distance(&cpdag2);
        assert_eq!(distance, 5); // Edges differ
    }

    #[test]
    fn test_hamming_distance_6_nodes() {
        let network1: HashMap<u8, Vec<u8>> = vec![
            (2, vec![1]),   // 1 -> 2
            (3, vec![1, 2]),// 1 -> 3, 2 -> 3
            (4, vec![3]),   // 3 -> 4
            (5, vec![4]),   // 4 -> 5
            (6, vec![5]),   // 5 -> 6
        ].into_iter().collect();

        let network2: HashMap<u8, Vec<u8>> = vec![
            (2, vec![1]),   // 1 -> 2
            (3, vec![2]),   // 2 -> 3
            (4, vec![1, 3]),// 1 -> 4, 3 -> 4
            (5, vec![1]),   // 1 -> 5
            (6, vec![4]),   // 4 -> 6
        ].into_iter().collect();

        let cpdag1 = create_cpdag(network1);
        let cpdag2 = create_cpdag(network2);
        println!("{:?}", cpdag1);
        println!("{:?}", cpdag2);
        let distance = cpdag1.hamming_distance(&cpdag2);
        assert_eq!(distance, 7); // Edges differ
    }

    #[test]
    fn test_hamming_distance_7_nodes() {
        let network1: HashMap<u8, Vec<u8>> = vec![
            (2, vec![1]),   // 1 -> 2
            (3, vec![1, 2]),// 1 -> 3, 2 -> 3
            (4, vec![3]),   // 3 -> 4
            (5, vec![4]),   // 4 -> 5
            (6, vec![5]),   // 5 -> 6
            (7, vec![6]),   // 6 -> 7
        ].into_iter().collect();

        let network2: HashMap<u8, Vec<u8>> = vec![
            (2, vec![1]),   // 1 -> 2
            (3, vec![2]),   // 2 -> 3
            (4, vec![1, 3]),// 1 -> 4, 3 -> 4
            (5, vec![1]),   // 1 -> 5
            (6, vec![4]),   // 4 -> 6
            (7, vec![3]),   // 3 -> 7
        ].into_iter().collect();

        let cpdag1 = create_cpdag(network1);
        let cpdag2 = create_cpdag(network2);
        println!("{:?}", cpdag1);
        println!("{:?}", cpdag2);
        let distance = cpdag1.hamming_distance(&cpdag2);
        assert_eq!(distance, 11);// 9 // Edges differ
    }
}