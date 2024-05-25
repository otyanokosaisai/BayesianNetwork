use std::{collections::HashMap, fs::File, io::{self, BufRead, BufReader, Read}, path::Path};

use petgraph::graph::DiGraph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Setting {
    pub method: String,
    pub bdeu_ess: f64,
    pub data_path: String,
    pub compare_network_path: String,
    pub saving_dir: String,
}

pub fn read_settings_from_file(file_path: &str) -> Result<Setting, Box<dyn std::error::Error>> {
    // ファイルを開く
    let mut file = File::open(file_path)?;
    // ファイル内容を文字列に読み込む
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    // YAML文字列を構造体にデシリアライズする
    let settings: Setting = serde_yaml::from_str(&contents)?;
    Ok(settings)
}

pub fn read_dot_file<P: AsRef<Path>>(file_path: P, header: &HashMap<&str, u8>) -> io::Result<DiGraph<u8, ()>> {
    // println!("[read_dot_file] header: {:?}", header);
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();

    // ノードを追加する（インデックスはヘッダーマップに基づく）
    let mut node_indices: HashMap<&str, petgraph::prelude::NodeIndex> = HashMap::new();
    let mut header_vec: Vec<(&str, u8)> = header.iter().map(|(k, v)| (*k, *v)).collect();
    header_vec.sort_by(|a, b| a.1.cmp(&b.1));
    for (key, index) in header_vec {
        let node_index = graph.add_node(index);
        // println!("Adding node: {:?} -> {:?}", key, node_index); // デバッグ用出力
        node_indices.insert(key, node_index);
    }

    for line in reader.lines() {
        let line = line?;
        // 行をトリムして余分なスペースやセミコロンを削除
        let line = line.trim().trim_end_matches(';').trim();
        if let Some((source, target)) = line.split_once(" -> ") {
            let source = source.trim_matches('"');
            let target = target.trim_matches('"');

            // println!("Processing edge: {} -> {}", source, target); // デバッグ用出力

            if let (Some(&source_index), Some(&target_index)) = (node_indices.get(source), node_indices.get(target)) {
                // println!("Adding edge: {:?} -> {:?}", source_index, target_index); // デバッグ用出力
                graph.add_edge(target_index, source_index , ());
            }
        }
    }
    Ok(graph)
}