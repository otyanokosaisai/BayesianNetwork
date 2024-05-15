use chrono::Local;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::fs;

pub fn render_graph_to_dot(
    graph: &HashMap<u8, HashSet<u8>>, 
    headers_index_map: &HashMap<u8, String>
) -> Result<(), std::io::Error> {
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
