
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::sync::Arc;
use std::collections::HashMap;
use std::thread;
use chrono::Local;
use csv::ReaderBuilder;
use dashmap::DashMap;
use itertools::Itertools;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use std::error::Error;
use anyhow::Result;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::fmt::Write;

#[derive(Debug)]
pub struct DataContainer {
    pub ct: CrossTable,
    pub cft: CrossFrequencyTable,
    pub ls: LocalScores,
    pub sample_size: u32,
    pub best_parents: BestParents,
    pub sinks: Sinks,
    pub order: Order,
    pub network: Network,
    pub compare_network: Network,
}

#[derive(Debug)]
pub struct CrossTable {
    pub ct_values: DashMap<Vec<u8>, u32>,
    pub header: HashMap<u8, String>,
    pub category_maps: HashMap<u8, HashMap<String, u8>>,
}
#[derive(Debug)]
pub struct CrossFrequencyTable {
    pub cft_values: DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u32>>>,
}
#[derive(Debug)]
pub struct LocalScores {
    pub ls_values: DashMap<u8, DashMap<Vec<u8>, f32>>,
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


pub fn load_data(path: &str) -> Result<DataContainer, Box<dyn Error>> {
    let file = File::open(path)?;
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
    let chunk_size = (data.len() + num_cpus - 1) / num_cpus;

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

    let headers_index_map: HashMap<u8, String> = headers.iter().enumerate().map(|(i, h)| (i as u8, h.clone())).collect();
    let final_category_maps: HashMap<u8, HashMap<String, u8>> = category_maps.into_iter().enumerate()
        .map(|(i, map)| (i as u8, map))
        .collect();
    let sample_size = freq_map.iter().map(|entry| *entry.value()).sum();
    Ok(DataContainer {
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
    })
}

impl DataContainer {
    pub fn ct2cft(&mut self) {
        self.cft.initialize(&self.ct);
    }
    pub fn _get_local_scores(&mut self, method: &str) {
        self.ls.initialize(&self.cft, method, self.sample_size);
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
    pub fn analyze(&mut self, method: &str) {
        self.ct2cft();
        self._get_local_scores(method);
        self._get_best_parents();
        self._get_best_sinks();
        self._get_order();
        self._get_network();
    }
    pub fn evaluate(&self) -> f64 {
        self.network.calculate_bic(&self.cft, self.sample_size as usize)
    }
    pub fn visualize(&self) {
        match self.network.render_graph_to_dot(&self.ct.header) {
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
        let cft: DashMap<u8, DashMap<Vec<u8>, HashMap<u8, u32>>> = DashMap::new(); // child index, parent values, child value, count

        all_valiables.par_iter().for_each(|&child_valiable_index| {
            let parent_indices: Vec<u8> = all_valiables.iter().filter(|&&v| v != child_valiable_index).map(|&v| v).collect();
            let cft_sub: DashMap<Vec<u8>, HashMap<u8, u32>> = DashMap::new();
            
            self.ct_values.iter().for_each(|entry| {
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
        CrossFrequencyTable {
            cft_values: cft,
        }
    }
}

impl CrossFrequencyTable {
    pub fn initialize(&mut self, ct: &CrossTable) {
        self.cft_values = ct.ct2cft().cft_values;
    }
    //method : aic or bic
    pub fn get_local_scores(&self, method: &str, sample_size: u32) -> DashMap<u8, DashMap<Vec<u8>, f32>> {
        match method {
            "bic" => {
                self.bic(sample_size)
            },
            // "aic" => {
            //     // self.aic();
            //     Err("aic is not implemented".to_string());
            // },
            // "bdeu"
            _ => {
                panic!("method must be 'aic' or 'bic'");
            }
        }
    }
    pub fn bic(&self, sample_size: u32) -> DashMap<u8, DashMap<Vec<u8>, f32>> {
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
            calculate_bic_recursive(&mut parent_indices, &data_map, sample_size, &mut local_scores_for_child, &mut visited);
            local_scores.insert(child, local_scores_for_child);
        });
        local_scores
    }
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

impl LocalScores {
    pub fn initialize(&mut self, cft: &CrossFrequencyTable, method: &str, sample_size: u32) {
        self.ls_values = cft.get_local_scores(method, sample_size);
    }
    pub fn get(&self, child: u8, parents_candidates: &Vec<u8>) -> Result<f32, String> {
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
        let mut scores: HashMap<Vec<u8>, f32> = HashMap::new();
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
        // println!("ord: {:?}", ord);
        Order {
            order_values: ord,
        }
    }
}

impl Order {
    pub fn initialize(&mut self, variables: Vec<u8>, sinks: &Sinks) {
        self.order_values = sinks.sinks2order(variables).order_values;
    }
    pub fn ord2net(&self, variables: Vec<u8>, best_parents: &BestParents) -> Network {
        let mut parents: Vec<Vec<u8>> = vec![vec![]; variables.len()];
        let mut predecs: Vec<u8> = Vec::new();
        for (child_index, &child) in self.order_values.iter().enumerate() {
            let ord_i = child;
            predecs.sort();
            // let ls = local_scores.get(child, &predecs).expect("REASON");
            // let bps_value = match get_best_parents(ord_i, &predecs, &ls).get(&predecs) {
            let bps_value = best_parents.get(child, &predecs).expect("REASON");
            parents[ord_i as usize] = bps_value.clone(); // child -> ord_i に変更
            // parents[child] = bps_value.clone();
            predecs.push(ord_i);
            // println!("{:?} {:?}", ord_i, bps_value);
        }
        Network {
            network_values: parents.into_iter().enumerate().map(|(i, p)| (i as u8, p.into_iter().collect())).collect()
        }
    }
}

impl Network {
    pub fn initialize(&mut self, variables: Vec<u8>, order: &Order, best_parents: &BestParents) {
        self.network_values = order.ord2net(variables, best_parents).network_values;
    }
    pub fn calculate_bic(
        &self,
        cft: &CrossFrequencyTable,
        total_samples: usize,
    ) -> f64 {
        let graph = &self.network_values;
        let cft = &cft.cft_values;
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
    pub fn render_graph_to_dot(
        &self,
        headers_index_map: &HashMap<u8, String>
    ) -> Result<(), std::io::Error> {
        let graph = &self.network_values;
        let mut dot_output = String::from("digraph G {\n");
        
        let unknown = "?".to_string();
        // グラフのエッジを処理
        for (source, targets) in graph {
            let source_name = headers_index_map.get(source).unwrap_or(&unknown);
            for target in targets {
                let target_name = headers_index_map.get(target).unwrap_or(&unknown);
                writeln!(&mut dot_output, "    \"{}\" -> \"{}\";", target_name, source_name).unwrap();
                // writeln!(&mut dot_output, "    \"{}\" -> \"{}\";",source_name, target_name).unwrap();
            }
        }
    
        dot_output.push_str("}\n");
    
        // 現在の日時を取得してファイル名を生成
        let now = Local::now();
        let filename = format!("./figures/graph{}.dot", now.format("%Y%m%d%H%M%S"));
    
        // ディレクトリの存在を確認し、存在しない場合は作成
        fs::create_dir_all("./figures")?;
    
        // DOT データをファイルに書き込む
        fs::write(&filename, dot_output)?;
    
        // 成功した場合のメッセージ出力
        println!("Run 'dot -Tpng {} > {}.png' to generate a PNG file", filename, filename);
        Ok(())
    }
}