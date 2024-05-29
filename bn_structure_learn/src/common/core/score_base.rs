use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use std::thread;
use chrono::Local;
use csv::ReaderBuilder;
use dashmap::DashMap;
use itertools::Itertools;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use std::error::Error;
use anyhow::Result;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::fmt::Write;
use statrs::function::gamma::ln_gamma;
use crate::common::core::utils::{Setting, read_settings_from_file, read_dot_file};


// fn read_settings_from_file(file_path: &str) -> Result<Setting, Box<dyn std::error::Error>> {
//     // ファイルを開く
//     let mut file = File::open(file_path)?;
//     // ファイル内容を文字列に読み込む
//     let mut contents = String::new();
//     file.read_to_string(&mut contents)?;

//     // YAML文字列を構造体にデシリアライズする
//     let settings: Setting = serde_yaml::from_str(&contents)?;
//     Ok(settings)
// }

// fn read_dot_file<P: AsRef<Path>>(file_path: P, header: &HashMap<&str, u8>) -> io::Result<DiGraph<u8, ()>> {
//     // println!("[read_dot_file] header: {:?}", header);
//     let file = File::open(file_path)?;
//     let reader = BufReader::new(file);
//     let mut graph = DiGraph::new();

//     // ノードを追加する（インデックスはヘッダーマップに基づく）
//     let mut node_indices: HashMap<&str, petgraph::prelude::NodeIndex> = HashMap::new();
//     let mut header_vec: Vec<(&str, u8)> = header.iter().map(|(k, v)| (*k, *v)).collect();
//     header_vec.sort_by(|a, b| a.1.cmp(&b.1));
//     for (key, index) in header_vec {
//         let node_index = graph.add_node(index);
//         // println!("Adding node: {:?} -> {:?}", key, node_index); // デバッグ用出力
//         node_indices.insert(key, node_index);
//     }

//     for line in reader.lines() {
//         let line = line?;
//         // 行をトリムして余分なスペースやセミコロンを削除
//         let line = line.trim().trim_end_matches(';').trim();
//         if let Some((source, target)) = line.split_once(" -> ") {
//             let source = source.trim_matches('"');
//             let target = target.trim_matches('"');

//             // println!("Processing edge: {} -> {}", source, target); // デバッグ用出力

//             if let (Some(&source_index), Some(&target_index)) = (node_indices.get(source), node_indices.get(target)) {
//                 // println!("Adding edge: {:?} -> {:?}", source_index, target_index); // デバッグ用出力
//                 graph.add_edge(target_index, source_index , ());
//             }
//         }
//     }
//     Ok(graph)
// }

fn build_network_from_graph(graph: DiGraph<u8, ()>) -> Network {
    // println!("[load_graph] {:?}", graph);
    let mut network_values: HashMap<u8, HashSet<u8>> = HashMap::new();

    for edge in graph.edge_references() {
        let source = edge.source().index() as u8;
        let target = edge.target().index() as u8;

        network_values
            .entry(source)
            .or_insert_with(HashSet::new)
            .insert(target);
    }
    // エッジにおいて存在しないノードを親ナシノードとして追加   
    for node in graph.node_indices() {
        let node = node.index() as u8;
        if !network_values.contains_key(&node) {
            network_values.insert(node, HashSet::new());
        }
    }

    Network { network_values }
}

pub fn load_data(setting_path: &str) -> Result<DataContainer, Box<dyn Error>> {
    let setting = read_settings_from_file(setting_path)?;
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

    let data = Arc::new(data);
    let freq_map = Arc::new(DashMap::new());
    let num_cpus = num_cpus::get();
    let chunk_size = (data.len() + num_cpus - 1) / num_cpus; // / 40000
    let handles: Vec<_> = (0..num_cpus).map(|i| {
        let data = Arc::clone(&data);
        let freq_map = Arc::clone(&freq_map);
        thread::spawn(move || {
            let start = i * chunk_size;
            let end = std::cmp::min((i + 1) * chunk_size, data.len());
            for row in &data[start..end] {
                freq_map.entry(row.clone()).and_modify(|e| *e += 1).or_insert(1);
            }
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let headers_index_map: HashMap<u8, String> = headers.iter().enumerate().map(|(i, h)| (i as u8, h.to_string())).collect();
    let final_category_maps: HashMap<u8, HashMap<String, u8>> = category_maps.into_iter().enumerate()
        .map(|(i, map)| (i as u8, map))
        .collect();
    let sample_size = freq_map.iter().map(|entry| *entry.value()).sum();
    
    // compare_network_pathが見つからない場合はcompare_networkを空にする
    if !Path::new(&setting.compare_network_path).exists() {
        return Ok(DataContainer {
            setting: setting.clone(),
            ct: CrossTable {
                ct_values: freq_map.as_ref().clone(),
                header: headers_index_map,
                category_maps: final_category_maps,
            },
            cft: CrossFrequencyTable {
                cft_values: DashMap::new(),
            },
            ls: LocalScores {
                ls_values: DashMap::new(),
            },
            sample_size,
            best_parents: BestParents {
                bp_values: HashMap::new(),
            },
            sinks: Sinks {
                sinks_values: HashMap::new(),
            },
            order: Order {
                order_values: Vec::new(),
            },
            network: Network {
                network_values: HashMap::new(),
            },
            scoring_method: ScoringMethod {
                method: ScoringMethodName::None,
            },
            compare_network: Network {
                network_values: HashMap::new(),
            },
        });
    }
    let headers_tmp: HashMap<&str, u8> = headers_index_map.iter().map(|(k, v)| (v.as_str(), *k)).collect();
    let dot_graph = read_dot_file(&setting.compare_network_path, &headers_tmp)?;
    let network = build_network_from_graph(dot_graph);
    Ok(DataContainer {
        setting: setting.clone(),
        ct: CrossTable {
            ct_values: freq_map.as_ref().clone(),
            header: headers_index_map,
            category_maps: final_category_maps,
        },
        cft: CrossFrequencyTable {
            cft_values: DashMap::new(),
        },
        ls: LocalScores {
            ls_values: DashMap::new(),
        },
        sample_size,
        best_parents: BestParents {
            bp_values: HashMap::new(),
        },
        sinks: Sinks {
            sinks_values: HashMap::new(),
        },
        order: Order {
            order_values: Vec::new(),
        },
        network: Network {
            network_values: HashMap::new(),
        },
        scoring_method: ScoringMethod {
            method: ScoringMethodName::None,
        },
        compare_network: network,
    })
}

pub fn load_data_exp(setting: Setting) -> Result<DataContainer, Box<dyn Error>> {
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

    let data = Arc::new(data);
    let freq_map = Arc::new(DashMap::new());
    let num_cpus = num_cpus::get();
    let chunk_size = (data.len() + num_cpus - 1) / num_cpus; // / 40000
    let handles: Vec<_> = (0..num_cpus).map(|i| {
        let data = Arc::clone(&data);
        let freq_map = Arc::clone(&freq_map);
        thread::spawn(move || {
            let start = i * chunk_size;
            let end = std::cmp::min((i + 1) * chunk_size, data.len());
            for row in &data[start..end] {
                freq_map.entry(row.clone()).and_modify(|e| *e += 1).or_insert(1);
            }
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let headers_index_map: HashMap<u8, String> = headers.iter().enumerate().map(|(i, h)| (i as u8, h.to_string())).collect();
    let final_category_maps: HashMap<u8, HashMap<String, u8>> = category_maps.into_iter().enumerate()
        .map(|(i, map)| (i as u8, map))
        .collect();
    let sample_size = freq_map.iter().map(|entry| *entry.value()).sum();
    let headers_tmp: HashMap<&str, u8> = headers_index_map.iter().map(|(k, v)| (v.as_str(), *k)).collect();
    let dot_graph = read_dot_file(&setting.compare_network_path, &headers_tmp)?;
    let network = build_network_from_graph(dot_graph);
    Ok(DataContainer {
        setting: setting.clone(),
        ct: CrossTable {
            ct_values: freq_map.as_ref().clone(),
            header: headers_index_map,
            category_maps: final_category_maps,
        },
        cft: CrossFrequencyTable {
            cft_values: DashMap::new(),
        },
        ls: LocalScores {
            ls_values: DashMap::new(),
        },
        sample_size,
        best_parents: BestParents {
            bp_values: HashMap::new(),
        },
        sinks: Sinks {
            sinks_values: HashMap::new(),
        },
        order: Order {
            order_values: Vec::new(),
        },
        network: Network {
            network_values: HashMap::new(),
        },
        scoring_method: ScoringMethod {
            method: ScoringMethodName::None,
        },
        compare_network: network,
    })
}

#[derive(Debug)]
pub struct DataContainer {
    pub setting: Setting,
    pub ct: CrossTable,
    pub cft: CrossFrequencyTable,
    pub ls: LocalScores,
    pub sample_size: u64,
    pub best_parents: BestParents,
    pub sinks: Sinks,
    pub order: Order,
    pub network: Network,
    pub scoring_method: ScoringMethod,
    pub compare_network: Network,
}

#[derive(Debug)]
pub struct CrossTable {
    pub ct_values: DashMap<Vec<u8>, u64>,
    pub header: HashMap<u8, String>,
    pub category_maps: HashMap<u8, HashMap<String, u8>>,
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
pub struct BestParents {
    pub bp_values: HashMap<u8, HashMap<Vec<u8>, Vec<u8>>>,
}
#[derive(Debug)]
pub struct Sinks {
    pub sinks_values: HashMap<Vec<u8>, u8>,
}
#[derive(Debug)]
pub struct Order {
    pub order_values: Vec<u8>,
}
#[derive(Debug)]
pub struct Network {
    pub network_values: HashMap<u8, HashSet<u8>>,
}

#[derive(Debug)]
pub enum ScoringMethodName {
    Bic,
    Aic,
    BDeu(f64),
    None,
}

impl ScoringMethodName {
    pub fn from_string(method: &str, ess: f64) -> ScoringMethodName {
        match method {
            "bic" => ScoringMethodName::Bic,
            "aic" => ScoringMethodName::Aic,
            "bdeu" => ScoringMethodName::BDeu(ess),
            _ => panic!("method must be 'bic' or 'aic'"),
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
        indices: &mut Vec<u8>,
        data_map: &DashMap<Vec<u8>, HashMap<u8, u64>>,
        sample_size: u64,
        local_scores_for_child: &DashMap<Vec<u8>, f64>,
        visited: &HashSet<Vec<u8>>,
        category_map: &HashMap<u8, u8>,
    ) {
        if visited.contains(indices) {
            return;
        }
        let mut visited_clone = visited.clone();
        visited_clone.insert(indices.to_vec());
        let bic = self.calculate_for_parent_set(&data_map, sample_size as usize, indices, child, category_map);
        local_scores_for_child.insert(indices.to_vec(), bic);
        (0..indices.len()).into_par_iter().for_each(|index| {
            let mut new_indices = indices.clone();
            new_indices.remove(index);
            let new_data_for_parent_subset: DashMap<Vec<u8>, HashMap<u8, u64>> = DashMap::new();
            data_map.iter().for_each(|entry| {
                let mut key = entry.key().clone();
                let value_map = entry.value();  
                key.remove(index);
                new_data_for_parent_subset.entry(key).and_modify(|existing_map: &mut HashMap<u8, u64>| {
                    for (key, value) in value_map.iter() {
                        *existing_map.entry(*key).or_insert(0) += value;
                    }
                }).or_insert_with(|| value_map.clone());
            });
            self.calculate_recursive(child, &mut new_indices, &new_data_for_parent_subset, sample_size, local_scores_for_child, visited, category_map);
        });
    }
        
    fn calculate_for_parent_set(&self, parent_data: &DashMap<Vec<u8>, HashMap<u8, u64>>, sample_size: usize, parents: &Vec<u8>, child: u8, category_map: &HashMap<u8, u8>) -> f64 {
        let k: f64 = parents.iter().map(|&i| category_map[&i] as f64).product::<f64>() * (category_map[&child] as f64 - 1.0);
        let l: f64 = ScoringMethod::calculate_log_likelihood(parent_data);
        match self.method {
            ScoringMethodName::Bic => l - k * (sample_size as f64).ln() / 2.0,
            ScoringMethodName::Aic => l - k,
            ScoringMethodName::BDeu(ess) => ScoringMethod::calculate_bdeu_score(parent_data, parents, child, category_map, ess),
            ScoringMethodName::None => panic!("method must be 'bic' or 'aic' or 'bdeu'"),
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
        score
    }
}

impl DataContainer {
    pub fn ct2cft(&mut self) {
        self.cft.initialize(&self.ct);
    }
    pub fn _get_local_scores(&mut self) {
        let category_nums: HashMap<u8, u8> = self.ct.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect();
        self.ls.initialize(&self.cft, self.sample_size, &self.scoring_method, &category_nums);
    }
    pub fn _get_best_parents(&mut self) {
        self.best_parents.initialize((0..self.ct.header.len() as u8).collect(), &self.ls, (0..self.ct.header.len() as u8).collect());
    }
    pub fn _get_best_sinks(&mut self) {
        self.sinks = Sinks {
            sinks_values: HashMap::new(),
        };
        self.sinks.initialize((0..self.ct.header.len() as u8).collect(), &self.ls, &self.best_parents);
    }
    pub fn _get_order(&mut self) {
        self.order = Order {
            order_values: Vec::new(),
        };
        self.order.initialize((0..self.ct.header.len() as u8).collect(), &self.sinks);
    }
    pub fn _get_network(&mut self) {
        self.network = Network {
            network_values: HashMap::new(),
        };
        self.network.initialize((0..self.ct.header.len() as u8).collect(), &self.order, &self.best_parents);
    }
    pub fn analyze(&mut self) {
        self.scoring_method.method = ScoringMethodName::from_string(&self.setting.method, self.setting.bdeu_ess);
        self.ct2cft();
        self._get_local_scores();
        self._get_best_parents();
        self._get_best_sinks();
        self._get_order();
        self._get_network();
    }
    pub fn evaluate(&self) -> f64 {
        // let category_nums = self.ct.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect();
        let category_nums: HashMap<u8, u8> = self.ct.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect();
        let score = self.network.evaluate(&self.cft, self.sample_size as usize, &self.scoring_method, &category_nums);
        score
    }
    pub fn compare(&self) {
        let optimized_score = self.evaluate();
        let category_nums: HashMap<u8, u8> = self.ct.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect();
        let score = self.compare_network.evaluate(&self.cft, self.sample_size as usize, &self.scoring_method, &category_nums);
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
        let category_nums: HashMap<u8, u8> = self.ct.category_maps.iter().map(|(i, map)| (*i, map.len() as u8)).collect();
        let score = self.compare_network.evaluate(&self.cft, self.sample_size as usize, &self.scoring_method, &category_nums);
        output.push(format!("[Aic] optimized_score: {:?}, ans_score: {:?}", optimized_score, score));

        self.scoring_method.method = ScoringMethodName::Bic;
        let optimized_score = self.evaluate();
        let score = self.compare_network.evaluate(&self.cft, self.sample_size as usize, &self.scoring_method, &category_nums);
        output.push(format!("[Bic] optimized_score: {:?}, ans_score: {:?}", optimized_score, score));

        self.scoring_method.method = ScoringMethodName::BDeu(self.setting.bdeu_ess);
        let optimized_score = self.evaluate();
        let score = self.compare_network.evaluate(&self.cft, self.sample_size as usize, &self.scoring_method, &category_nums);
        output.push(format!("[BDeu({:?})] optimized_score: {:?}, ans_score: {:?}", self.setting.bdeu_ess, optimized_score, score));
    
        let cpdag1 = self.network.to_cpdag();
        let cpdag2 = self.compare_network.to_cpdag();
        let hamming_distance = cpdag1.hamming_distance(&cpdag2);
        output.push(format!("[hamming_distance]: {:?}", hamming_distance));
    
        output
    }
    pub fn visualize(&self) {
        match self.network.render_graph_to_dot(&self.ct.header, &self.setting.saving_dir) {
            Ok(_) => {
            },
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}

impl CrossTable {
    pub fn ct2cft(&self) -> CrossFrequencyTable {
        let num_valiables = self.header.len() as u8;
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
    pub fn initialize(&mut self, ct: &CrossTable) {
        self.cft_values = ct.ct2cft().cft_values;
    }
    pub fn get_local_scores(&self, sample_size: u64, scoring_method: &ScoringMethod, category_map: &HashMap<u8, u8>) -> LocalScores {
        LocalScores {
            ls_values: self.bic(sample_size, scoring_method, category_map),
        }
    }
    pub fn bic(&self, sample_size: u64, scoring_method: &ScoringMethod, category_map: &HashMap<u8, u8>) -> DashMap<u8, DashMap<Vec<u8>, f64>> {
        let data = &self.cft_values;
        let local_scores = DashMap::new();
        let variables: Vec<u8> = data.iter().map(|entry| *entry.key()).collect();
        variables.clone().into_par_iter().enumerate().for_each(|(child_index, child)| {
            let mut parent_indices = variables.clone();
            parent_indices.remove(child_index);
            parent_indices.sort();
            let data_map = data.get(&child).unwrap().clone();
            let mut local_scores_for_child = DashMap::new();
            let mut visited = HashSet::new();
            // scoring_method.calculate_recursive(&mut parent_indices, &data_map, sample_size, &mut local_scores_for_child, &mut visited);
            scoring_method.calculate_recursive(child,&mut parent_indices, &data_map, sample_size, &mut local_scores_for_child, &mut visited,category_map);
            // println!("{:?}", sample_size);
            local_scores.insert(child, local_scores_for_child);
        });
        local_scores
    }
}



impl LocalScores {
    pub fn initialize(&mut self, cft: &CrossFrequencyTable,  sample_size: u64, scoring_method: &ScoringMethod, category_map: &HashMap<u8, u8>) {
        self.ls_values = cft.get_local_scores(sample_size, scoring_method, category_map).ls_values;
    }
    pub fn get(&self, child: u8, parents_candidates: &Vec<u8>) -> Result<f64, String> {
        let ls_v = self.ls_values.get(&child).ok_or("child not found".to_string())?;
        let local_score = ls_v.get(parents_candidates).ok_or("parents not found".to_string())?;
        Ok(*local_score)
    }
    pub fn _get_best_parents(&self, child: u8, parents_variables: Vec<u8>) -> Result<HashMap<Vec<u8>, Vec<u8>>, String> {
        let mut best_scores = HashMap::new();
        let mut best_parent_sets = HashMap::new();
        for cs in parents_variables.iter().copied().powerset() {
            let cs_cloned = cs.clone();
            let score = match self.get(child,&cs) {
                Ok(score) => score,
                Err(e) => return Err(e),
            };
            best_scores.insert(cs_cloned.clone(), score);  // Use the cloned version in the HashMap.
            best_parent_sets.insert(cs_cloned.clone(), cs_cloned.clone());
            for j in 0..cs_cloned.len() {
                let mut cs1 = cs_cloned.clone();
                cs1.remove(j);
                let score = best_scores.get(&cs1).unwrap();
                if *score > *best_scores.get(&cs_cloned).unwrap() {
                    // if cs1.len() == 0 {
                    //     println!("[] was considered and {:?}(:[]) > {:?}", *score, *best_scores.get(&cs_cloned).unwrap());
                    // }
                    best_scores.insert(cs_cloned.clone(), *score);
                    best_parent_sets.insert(cs_cloned.clone(), best_parent_sets.get(&cs1).unwrap().clone());
                }
            }
        }
        Ok(best_parent_sets)
    }
}

impl BestParents {
    pub fn initialize(&mut self, v: Vec<u8>, local_scores: &LocalScores, variables: Vec<u8>) {
        let mut best_parents: HashMap<u8, HashMap<Vec<u8>, Vec<u8>>> = HashMap::new();
        for &child in v.iter() {
            let parents_candidates: Vec<u8> = variables.iter().filter(|&v| *v != child).map(|&v| v).collect();
            let best_parents_for_child = local_scores._get_best_parents(child, parents_candidates).unwrap();
            best_parents.insert(child, best_parents_for_child);
        }
        self.bp_values = best_parents;        
    }
    pub fn get(&self, child: u8, parents_candidates: &Vec<u8>) -> Result<&Vec<u8>, String> {
        self.bp_values.get(&child).expect("REASON").get(parents_candidates).ok_or("child not found".to_string())
    }
    pub fn get_best_sinks(&self, variables: Vec<u8>, local_scores: &LocalScores) -> Sinks {
        let mut scores: HashMap<Vec<u8>, f64> = HashMap::new();
        let mut sinks: HashMap<Vec<u8>, u8> = HashMap::new();
        let subsets = variables.iter()
            .flat_map(|size| variables.iter().copied().combinations(*size as usize))
            .chain(std::iter::once(variables.clone()))
            .map(|combo| combo.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for w in subsets {
            scores.insert(w.clone(), 0.0);
            sinks.insert(w.clone(), u8::MAX);  // u8::MAX を仮の "無効" 値として使用
            for (index, &sink) in w.iter().enumerate() {
                let mut upvars = w.clone();
                upvars.remove(index);  // sinkを除外
                let bps = self.get(sink, &upvars).unwrap();
                let skore = scores.get(&upvars).unwrap() 
                    + local_scores.get(sink, bps).unwrap();
                if sinks[&w] == u8::MAX || skore > *scores.get(&w).unwrap() {
                    scores.insert(w.clone(), skore);
                    sinks.insert(w.clone(), sink);
                }
            }
        }
        Sinks {
            sinks_values: sinks,
        }
    }
}
impl Sinks {
    pub fn initialize(&mut self, variables: Vec<u8>, local_scores: &LocalScores, best_parents: &BestParents) {
        self.sinks_values = best_parents.get_best_sinks(variables, local_scores).sinks_values;
    }
    pub fn get(&self, all_variables: &Vec<u8>) -> Result<u8, String> {
        self.sinks_values.get(all_variables).ok_or("all_variables not found".to_string()).map(|&v| v)
    }
    pub fn sinks2order(&self, variables: Vec<u8>) -> Order {
        let mut ord = vec![u8::MAX; variables.len()]; // ord 配列を初期化
        let mut left = variables.clone(); // left を v のクローンとして初期化
        left.sort();
        for i in (0..variables.len()).rev() {
            let current_sink = match self.get(&left) {
                Ok(sink) => sink,
                Err(e) => panic!("{}", e),
            }; // left に対応する最良の親ノードを取得
            ord[i] = current_sink;
            left.retain(|&x| x != current_sink); // left から選ばれた要素を削除
        }
        Order {
            order_values: ord,
        }
    }
}

impl Order {
    pub fn initialize(&mut self, variables: Vec<u8>, sinks: &Sinks) {
        self.order_values = sinks.sinks2order(variables).order_values;
    }
    pub fn ord2net(&self, variables: Vec<u8>, best_parents: &BestParents) -> Result<Network, String> {
        let mut parents: Vec<Vec<u8>> = vec![vec![]; variables.len()];
        let mut predecs: Vec<u8> = Vec::new();
        for &child in self.order_values.iter() {
            let ord_i = child;
            predecs.sort();
            let bps_value = best_parents.get(child, &predecs)?;
            parents[ord_i as usize] = bps_value.clone(); // child -> ord_i に変更
            predecs.push(ord_i);
        }
        Ok(Network {
            network_values: parents.into_iter().enumerate().map(|(i, p)| (i as u8, p.into_iter().collect())).collect()
        })
    }
}

impl Network {
    pub fn initialize(&mut self, variables: Vec<u8>, order: &Order, best_parents: &BestParents) {
        self.network_values = match order.ord2net(variables, best_parents) {
            Ok(net) => net.network_values,
            Err(e) => panic!("{}", e),
        };
    }
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

#[derive(Debug)]
pub struct CPDAG {
    #[allow(dead_code)]
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
        let all_edges: HashSet<Edge> = self.directed_edges.union(&self.undirected_edges).cloned().collect::<HashSet<Edge>>()
            .union(&other.directed_edges).cloned().collect::<HashSet<Edge>>()
            .union(&other.undirected_edges).cloned().collect::<HashSet<Edge>>();
        
        for edge in &all_edges {
            let self_has_directed = self.directed_edges.contains(edge);
            let other_has_directed = other.directed_edges.contains(edge);
            let reverse_edge = Edge { from: edge.to.clone(), to: edge.from.clone() };
            let self_has_undirected = self.undirected_edges.contains(edge) || self.undirected_edges.contains(&reverse_edge);
            let other_has_undirected = other.undirected_edges.contains(edge) || other.undirected_edges.contains(&reverse_edge);
            
            if self_has_directed != other_has_directed || self_has_undirected != other_has_undirected {
                distance += 1;
            }
        }

        distance
    }
}

pub fn create_cpdag(network: HashMap<u8, Vec<u8>>) -> CPDAG {
    // println!("[create_cpdag] {:?}", network);
    let nodes: Vec<Node> = network.keys().map(|&id| Node { id }).collect();
    let edges: Vec<Edge> = network.iter().flat_map(|(to, froms)| {
        froms.iter().map(move |&from| Edge { from: Node { id: from }, to: Node { id: *to } })
    }).collect();
    // println!("nodes: {:?}", nodes);
    // println!("edges: {:?}", edges);
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
    for (_, in_edges) in &incoming_edges {
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
                    }
                }
            }
        }
    }

    // 残りのエッジの処理
    for edge in &edges {
        if cpdag.is_marked_as_v_structure(edge) {
            cpdag.add_directed_edge(edge.clone());
        } else {
            let reverse_edge = Edge { from: edge.to.clone(), to: edge.from.clone() };
            if cpdag.is_marked_as_v_structure(&reverse_edge) {
                cpdag.add_directed_edge(edge.clone());
            } else {
                cpdag.add_undirected_edge(edge.clone());
            }
        }
    }  
    // println!("[CPDAG] {:?}", cpdag);

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